from datetime import datetime, timedelta
import logging
from io import StringIO
import re
from subprocess import PIPE, run
from typing import List, Tuple

from cachetools import cached, LRUCache, TTLCache
import numpy as np

from pandas import DataFrame, read_csv

fairshare_ttl_cache = TTLCache(maxsize=8, ttl=60)
squeue_ttl_cache = TTLCache(maxsize=8, ttl=60)
sinfo_ttl_cache = TTLCache(maxsize=8, ttl=60)
sstat_ttl_cache = TTLCache(maxsize=8, ttl=60)
sreport_topusers_cache = TTLCache(maxsize=8, ttl=60)
sreport_reservations_cache = TTLCache(maxsize=8, ttl=60)


@cached(cache=LRUCache(maxsize=512))
def dhhmmss_to_seconds(dhhmmss: str) -> int:
    days = 0
    try:
        if '-' in dhhmmss:
            days, dhhmmss = dhhmmss.split('-')
            days = int(days)
        while dhhmmss.count(':') < 2:
            dhhmmss = f'00:{dhhmmss}'

        h, m, s = map(int, dhhmmss.split(':'))
        seconds = ((days*24+h)*60+m)*60+s
        logging.debug(f'Converting "{ dhhmmss }" to seconds: { seconds }')
        return seconds
    except (ValueError, TypeError, AttributeError):
        logging.error(f'Error trying to convert { dhhmmss } to seconds. Returning 0.')
    return 0


@cached(cache=LRUCache(maxsize=512))
def seconds_to_hhmmss(seconds: int) -> str:
    return str(timedelta(seconds=int(seconds)))


@cached(cache=LRUCache(maxsize=1024))
def node_list_string_to_list(node_list: str) -> List[str]:
    nodes = []
    if node_list == "None assigned" or node_list == '(null)' or node_list is None:
        return nodes
    groups = re.findall(r'[^,\[]+(?:\[[^\]]+\])?', node_list)
    for group in groups:
        if '[' not in group:
            nodes.append(group)
            continue
        prefix = group.split('[')[0]
        node_ranges = group.split('[')[1][:-1].split(',')
        for node_range in node_ranges:
            if '-' not in node_range:
                nodes.append(f'{prefix}{node_range}')
                continue
            node_range = node_range.split('-')

            start, end = int(node_range[0]), int(node_range[1])
            zfill = len(node_range[0])
            for suffix in range(start, end + 1):
                nodes.append(f'{prefix}{str(suffix).zfill(zfill)}')
    return nodes


def run_command(command: str, parameters: list) -> Tuple[int, str, str]:
    logging.debug(f'Running command: { [command] + parameters }')
    cmd = run([command] + parameters, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    if cmd.returncode != 0:
        raise Exception(
            f'Error {cmd.returncode} running command: {command} {" ".join(parameters)}:\n'
            f'{cmd.stderr}'
        )
    return cmd.returncode, cmd.stdout.strip(), cmd.stderr.strip()


@cached(cache=squeue_ttl_cache)
def get_squeue() -> DataFrame:
    exit_status, stdout, stderr = run_command('squeue', ['-a', '--format=%u|%a|%A|%C|%D|%L|%M|%N|%T|%U'])
    squeue_data = read_csv(
        StringIO(stdout),
        sep='|',
        dtype={
            'STATE': 'object',
            'USER': 'object',
            'ACCOUNT': 'object',
            'JOBID': 'uint32',
            'CPUS': 'uint32',
            'NODES': 'uint16',
            'UID': 'uint32',
        },
        converters={
            'TIME': dhhmmss_to_seconds,
            'TIME_LEFT': dhhmmss_to_seconds,
            'NODELIST': node_list_string_to_list,
        }

    )

    squeue_data['CPUTIME_LEFT_SECONDS'] = squeue_data['TIME_LEFT'] * squeue_data['CPUS']

    logging.debug(f'squeue output: { squeue_data }')
    return squeue_data


@cached(cache=fairshare_ttl_cache)
def get_fairshare() -> DataFrame:
    exit_status, stdout, stderr = run_command('sshare', ['-a', '-l', '-P'])
    fairshare_data = read_csv(
        StringIO(stdout),
        sep='|',
        dtype={
            'RawShares': 'float32',
            'NormShares': 'float64',
            'RawUsage': 'uint64',
            'NormUsage': 'float64',
            'EffectvUsage': 'float64',
            'FairShare': 'float64',
            'LevelFS': 'float64',
        },
        converters={
            'Account': lambda x: x.strip(),
            'User': lambda x: x.strip() if x != '' else '-',
        },
        usecols=[
            'Account',
            'User',
            'RawShares',
            'NormShares',
            'RawUsage',
            'NormUsage',
            'EffectvUsage',
            'FairShare',
            'LevelFS',
        ],
        index_col=['Account', 'User'],
    )

    logging.debug(f'sshare output: {fairshare_data}')
    return fairshare_data


def megabytes_to_bytes_converter(megabytes: str) -> int:
    try:
        return int(megabytes) * (1024 ** 2)
    except ValueError:
        logging.debug(f'Could not convert "{megabytes}" to bytes unit integer')
        return 0


@cached(cache=sinfo_ttl_cache)
def get_sinfo() -> DataFrame:
    exit_status, stdout, stderr = run_command('sinfo', ['-N', '--format=%n|%e|%C|%m|%O|%R|%c|%T'])
    sinfo_data = read_csv(
        StringIO(stdout),
        sep='|',
        dtype={
            'CPU_LOAD': 'Float64',
            'CPUS': 'uint16',
            'PARTITION': 'object',
            'STATE': 'object',
        },
        converters={
            'FREE_MEM': megabytes_to_bytes_converter,
            'MEMORY': megabytes_to_bytes_converter,

        }
    )
    sinfo_data['FREE_MEM'] = sinfo_data[['FREE_MEM', 'MEMORY']].min(axis=1)
    sinfo_data['ALLOCATED_MEM'] = sinfo_data['MEMORY'] - sinfo_data['FREE_MEM']

    sinfo_data['allocated'] = sinfo_data['CPUS(A/I/O/T)'].apply(lambda x: int(x.split('/')[0]))
    sinfo_data['idle'] = sinfo_data['CPUS(A/I/O/T)'].apply(lambda x: int(x.split('/')[1]))
    sinfo_data['offline'] = sinfo_data['CPUS(A/I/O/T)'].apply(lambda x: int(x.split('/')[2]))
    sinfo_data['total'] = sinfo_data['CPUS(A/I/O/T)'].apply(lambda x: int(x.split('/')[3]))

    sinfo_data.drop(columns=['CPUS(A/I/O/T)'], inplace=True)
    sinfo_data = sinfo_data.drop_duplicates(subset=['HOSTNAMES', 'CPU_LOAD'])

    logging.debug(f'sinfo output: { sinfo_data }')

    return sinfo_data

@cached(cache=sreport_topusers_cache)
def get_top_users(count: int = 16, days: int = 3) -> DataFrame:
    now = datetime.now()
    time_format = '%Y-%m-%dT%H:%M:%S'
    end = now.strftime(time_format)
    start = (now - timedelta(days=days)).strftime(time_format)
    exit_status, stdout, stderr = run_command(
        'sreport',
        ['user', f'TopUsage', 'Group', f'TopCount={count}', f'Start={start}', f'End={end}', '-P'],
    )
    top_users_data = read_csv(
        StringIO(stdout),
        skiprows=[0, 1, 2, 3],
        sep='|',
        usecols=[
            'Login', 'Account', 'Used',
        ],
        index_col='Login',
        dtype={
            'Login': 'object',
            'User': 'uint32',
        },
        converters={
            'Account': lambda x: x.split(', ')
        }
    )
    return top_users_data


@cached(cache=sreport_reservations_cache)
def get_sreport_reservation():
    now = datetime.now()
    time_format = '%Y-%m-%dT%H:%M:%S'
    end = now.strftime(time_format)
    exit_status, stdout, stderr = run_command(
        'sreport',
        [
            'reservation',
            'Utilization',
            '-P',
            f'End={end}',
            'Format="Name,Start,End,Allocated,Idle,Nodes,TotalTime,TresName,TresCount,TresTime"',
        ],
    )
    reservation_data = read_csv(
        StringIO(stdout),
        skiprows=[0, 1, 2, 3],
        sep='|',
        usecols=[
            'Name', 'Start', 'End', 'TotalTime', 'Idle', 'Allocated', 'TRES Name', 'TRES count', 'TRES Time',
        ],
        index_col='Name',
        dtype={
            'Name': 'object',
            'Allocated': 'uint32',
            'Idle': 'uint32',
            'TRES Name': 'object',
            'TRES count': 'uint32',
        },
        converters={
            'Start': lambda x: datetime.strptime(x, time_format),
            'End': lambda x: datetime.strptime(x, time_format),
            'TRES Time': dhhmmss_to_seconds,
            'TotalTime': dhhmmss_to_seconds,
            'Nodes': lambda x: node_list_string_to_list,
        }
    )
    reservation_data = reservation_data.drop_duplicates(subset=['Name'])
    return reservation_data


@cached(cache=sstat_ttl_cache)
def get_sstat(squeue_data: DataFrame, users: list) -> DataFrame:
    running_jobs = squeue_data[squeue_data['STATE'] == 'RUNNING']
    if users != ['ALL']:
        running_jobs = running_jobs[running_jobs['USER'].isin(users)]

    job_ids = map(str, running_jobs['JOBID.1'].to_list())

    sstat_parameters = [
        '-j',
        ','.join(job_ids),
        '-P',
        '--format',
        'AveCPU,MaxRSS,JobID,AveDiskRead,AveDiskWrite,NTasks',
    ]

    exit_status, stdout, stderr = run_command('sstat', sstat_parameters)
    sstat_data = read_csv(StringIO(stdout), sep='|')
    sstat_data['JobID'] = sstat_data['JobID'].apply(lambda x: x.split('.')[0]).astype(int)
    sstat_data['AveCPU_SECONDS'] = sstat_data['AveCPU'].apply(lambda x: dhhmmss_to_seconds(x.split('.')[0]))
    logging.debug(f'sstat output:\n{ sstat_data }')
    sstat_data = squeue_data.merge(sstat_data, left_on='JOBID.1', right_on='JobID')
    return sstat_data


def grouped(iterable, n=2):
    """s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."""
    return zip(*[iter(iterable)]*n)


def create_account_cpu_usage_summary() -> dict:
    squeue_data = get_squeue().copy()

    final_dataframe = squeue_data.pivot_table(
        index='ACCOUNT',
        columns='STATE',
        values='CPUS',
        aggfunc=sum,
    )
    final_dataframe.fillna(0, inplace=True)
    logging.debug(final_dataframe.to_markdown())
    return {'account_cpu_usage_total': final_dataframe}


def create_account_cputime_remaining_summary() -> dict:
    squeue_data = get_squeue().copy()

    final_dataframe = squeue_data.pivot_table(
        index='ACCOUNT',
        columns='STATE',
        values='CPUTIME_LEFT_SECONDS',
        aggfunc=sum,
    )
    final_dataframe.fillna(0, inplace=True)
    logging.debug(final_dataframe.to_markdown())
    return {'account_cputime_remaining_total_seconds': final_dataframe}


# def create_job_detail_summary(job_detail: DataFrame, human_readable=True) -> DataFrame:
#     def calc_cpu_time(row):
#         cpu_max = row['TIME_SECONDS']*row['CPUS']
#         total_used = row['AveCPU_SECONDS']*row['NODES']
#         percentage_used = total_used / cpu_max if cpu_max != 0 else 0
#         return f'{naturaldelta(timedelta(seconds=total_used))} / {naturaldelta(timedelta(seconds=cpu_max))} ({int(percentage_used*100)}%)'
#
#     def calc_memory_used(row):
#         def mem_str_to_bytes(mem_str):
#             factor = {
#                 'B': 1,
#                 'K': 1024,
#                 'M': 1024**2,
#                 'G': 1024**3,
#                 'T': 1024**4,
#                 'P': 1024**5
#             }
#             try:
#                 return int(float(mem_str[:-1])*factor[mem_str[-1:]])
#             except TypeError:
#                 return 0
#
#         memory_allocated = mem_str_to_bytes(row['MIN_MEMORY']) * row['NODES']
#         memory_used = mem_str_to_bytes(row['MaxRSS']) * row['NODES']
#         percentage_used = memory_used / memory_allocated if memory_allocated != 0 else 0
#         if human_readable:
#             return f'{naturalsize(memory_used, binary=True)} / {naturalsize(memory_allocated, binary=True)} ({int(percentage_used*100)}%)'
#         return f'{memory_used} / {memory_allocated} ({int(percentage_used*100)}%)'
#
#     job_detail = job_detail.drop([
#         'ACCOUNT',
#         'JOBID.1',
#         'TIME_LEFT',
#         'NODELIST',
#         'STATE',
#         'UID',
#         'AveCPU',
#         'AveDiskRead',
#         'AveDiskWrite',
#     ], axis=1)
#
#     job_detail['cpu_used'] = job_detail.apply(calc_cpu_time, axis=1)
#     job_detail['memory_used'] = job_detail.apply(calc_memory_used, axis=1)
#
#     job_detail = job_detail.drop([
#         'TIME_SECONDS',
#         'TIME_LEFT_SECONDS',
#         'MaxRSS',
#         'NTasks',
#         'NODES',
#         'AveCPU_SECONDS',
#     ], axis=1)
#     job_detail.set_index('JobID', inplace=True)
#     return job_detail


def create_partition_memory_summary() -> dict:
    sinfo = get_sinfo().copy()
    logging.info(sinfo.columns)
    sinfo_mem = sinfo.groupby(['PARTITION']).agg({
        'FREE_MEM': ['sum'],
        'MEMORY': ['sum'],
    })
    sinfo_mem.columns = sinfo_mem.columns.get_level_values(0)


    sinfo_mem.rename(
        columns={
            'MEMORY': 'total',
            'FREE_MEM': 'free',
            'ALLOCATED_MEM': 'allocated',
        },
        inplace=True,
    )
    return {'memory_total_bytes': sinfo_mem}


def create_partition_node_state_summary():
    sinfo_data = get_sinfo().copy()
    node_states = sinfo_data.pivot_table(
            index='PARTITION',
            columns='STATE',
            values='HOSTNAMES',
            aggfunc='count',
        )
    node_states.fillna(0, inplace=True)
    return {
        'node_state_count': node_states,
    }


def create_partition_cpu_count_summary() -> dict:
    sinfo_cpu = get_sinfo().copy()

    sinfo_cpu = sinfo_cpu.groupby(['PARTITION']).agg({
        'allocated': ['sum'],
        'idle': ['sum'],
        'offline': ['sum'],
        'total': ['sum'],
    })
    sinfo_cpu.columns = sinfo_cpu.columns.get_level_values(0)
    return {'cpu_state_count': sinfo_cpu}


def create_partition_cpu_load_summary() -> dict:
    sinfo_cpu = get_sinfo().copy()

    sinfo_cpu = sinfo_cpu.groupby(['PARTITION']).agg({
        'CPU_LOAD': ['sum'],
        'allocated': ['sum'],
    })
    sinfo_cpu.columns = sinfo_cpu.columns.get_level_values(0)

    sinfo_cpu['load / allocated'] = sinfo_cpu['CPU_LOAD'].div(sinfo_cpu['allocated'])
    sinfo_cpu.rename(
        columns={
            'CPU_LOAD': 'load',
        },
        inplace=True,
    )

    return {'cpu_load': sinfo_cpu}


def create_fairshare_summaries() -> dict:
    fairshare = get_fairshare().copy()

    fairshare = fairshare[fairshare['RawShares'].notna()]
    fairshare = fairshare[fairshare['RawUsage'] > 60*60*24]  # filter out usage less than 24 hours
    logging.debug(fairshare)

    return {
        'fairshare_rawusage_seconds_total': fairshare['RawUsage'].to_frame(),
        'fairshare_normusage': fairshare['NormUsage'].to_frame(),
        'fairshare_levelfs': fairshare['LevelFS'].replace(np.inf, np.nan).dropna().to_frame(),
        'fairshare_rawshares': fairshare['RawShares'].to_frame(),
        'fairshare_normshares': fairshare['NormShares'].to_frame(),
    }


def create_node_summaries() -> dict:
    sinfo_data = get_sinfo()
    cpu_load = sinfo_data[['HOSTNAMES', 'CPU_LOAD']].copy()
    cpu_load.set_index('HOSTNAMES', inplace=True)
    cpu_load.index.rename('host', inplace=True)
    cpu_load.rename(
        columns={'CPU_LOAD': 'load'},
        inplace=True,
    )
    cpu_load.dropna(inplace=True)
    return {
        'host_cpu_load_percentage': cpu_load,
    }


def create_top_users_summaries() -> dict:
    days = 3
    top_users = get_top_users(count=16, days=3)
    top_users.index.rename('username', inplace=True)
    top_users.rename(
        columns={
            'Used': f'{days}-day_usage',
        },
        inplace=True,
    )
    top_users.drop('Account', axis='columns', inplace=True)
    return {
        'top_users_cpu_seconds_total': top_users,
    }


def create_reservation_summaries() -> dict:
    reservations = get_sreport_reservation().copy()

    return {
        'reservation_seconds_total': reservations[['Allocated', 'Idle']],
        'reservation_cpus_allocated_total': reservations[['TRES count']]
    }
