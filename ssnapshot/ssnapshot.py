#!/usr/bin/env python3

from datetime import timedelta
import logging
from io import StringIO
import re
from subprocess import PIPE, run
from typing import Tuple, List

from cachetools import cached, LRUCache, TTLCache
from humanize import naturalsize, naturaldelta
from pandas import DataFrame, read_csv


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


@cached(cache=LRUCache(maxsize=512))
def expand_compressed_slurm_nodelist(node_list) -> tuple:
    logging.debug(f'Expanding node list: { node_list }')
    if node_list == "None assigned" or node_list == '(null)' or node_list is None:
        return ()
    nodes = []
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
    return tuple(nodes)


def run_command(command: str, parameters: list) -> Tuple[int, str, str]:
    logging.debug(f'Running command: { command } { " ".join(parameters) }')
    cmd = run([command] + parameters, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    if cmd.returncode != 0:
        raise Exception(
            f'Error {cmd.returncode} running command: {command} {" ".join(parameters)}:\n'
            f'{cmd.stderr}'
        )
    return cmd.returncode, cmd.stdout.strip(), cmd.stderr.strip()


@cached(cache=TTLCache(maxsize=512, ttl=60))
def get_squeue() -> DataFrame:
    exit_status, stdout, stderr = run_command('squeue', ['-a', '--format=%all'])
    squeue_data = read_csv(StringIO(stdout), sep='|')
    logging.debug(f'squeue output: { squeue_data }')
    squeue_data = squeue_data.drop([
        'TRES_PER_NODE',
        'MIN_CPUS',
        'MIN_TMP_DISK',
        'FEATURES',
        'GROUP',
        'OVER_SUBSCRIBE',
        'COMMENT',
        'REQ_NODES',
        'COMMAND',
        'PRIORITY',
        'QOS',
        'REASON',
        'Unnamed: 18',
        'ST',
        'RESERVATION',
        'WCKEY',
        'EXC_NODES',
        'NODELIST(REASON)',
        'NICE',
        'S:C:T',
        # 'JOBID.1',
        'EXEC_HOST',
        'DEPENDENCY',
        'ARRAY_JOB_ID',
        'GROUP.1',
        'SOCKETS_PER_NODE',
        'CORES_PER_SOCKET',
        'THREADS_PER_CORE',
        'ARRAY_TASK_ID',
        'CONTIGUOUS',
        'PARTITION',
        'PRIORITY.1',
        # 'STATE',
        'LICENSES',
        'CORE_SPEC',
        'SCHEDNODES',
        'WORK_DIR',
        'NAME', 'SUBMIT_TIME', 'END_TIME', 'START_TIME', 'TIME_LIMIT',
    ], axis=1)
    for column in ['TIME_LEFT', 'TIME']:
        squeue_data[f'{column}_SECONDS'] = squeue_data[column].apply(dhhmmss_to_seconds)
    return squeue_data

@cached(cache=TTLCache(maxsize=512, ttl=60))
def get_sinfo() -> DataFrame:
    # exit_status, stdout, stderr = run_command('sinfo', ['-N', '--format=%all'])
    exit_status, stdout, stderr = run_command('sinfo', ['-N', '--format="%n|%e|%C|%m|%O|%R|%c"'])
    sinfo_data = read_csv(
        StringIO(stdout),
        sep='|',
        dtype={
            'CPU_LOAD': 'Float64',
            'CPUS': 'Int32',
            'FREE_MEM': 'Int32',
        }
    )
    # sinfo_data['CPU_LOAD'] = to_numeric(sinfo_data['CPU_LOAD'], errors='coerce')
    logging.debug(f'sinfo output: { sinfo_data }')

    return sinfo_data

@cached(cache=TTLCache(maxsize=512, ttl=60))
def get_sstat(squeue_data: DataFrame, users: list) -> DataFrame:
    running_jobs = squeue_data[squeue_data['STATE'] == 'RUNNING']
    if users != ['ALL']:
        running_jobs = running_jobs[running_jobs['USER'].isin(users)]

    job_ids = map(str, running_jobs['JOBID.1'].to_list())

    exit_status, stdout, stderr = run_command('sstat', ['-j', ','.join(job_ids), '-P', '--format', 'AveCPU,MaxRSS,JobID,AveDiskRead,AveDiskWrite,NTasks'])
    sstat_data = read_csv(StringIO(stdout), sep='|')
    sstat_data['JobID'] = sstat_data['JobID'].apply(lambda x: x.split('.')[0]).astype(int)
    sstat_data['AveCPU_SECONDS'] = sstat_data['AveCPU'].apply(lambda x: dhhmmss_to_seconds(x.split('.')[0]))
    logging.debug(f'sstat output:\n{ sstat_data }')
    sstat_data = squeue_data.merge(sstat_data, left_on='JOBID.1', right_on='JobID')
    return sstat_data


def grouped(iterable, n=2):
    """s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."""
    return zip(*[iter(iterable)]*n)


def create_job_summaries(squeue_data: DataFrame, human_readable: bool = True) -> Tuple[DataFrame, DataFrame]:

    squeue_data['CPUTIME_LEFT_SECONDS'] = squeue_data['TIME_LEFT_SECONDS'] * squeue_data['CPUS']

    squeue_grouped = squeue_data.groupby(['STATE', 'ACCOUNT'])  # , 'USER'])
    aggregated_grouped = squeue_grouped.agg({
        'CPUS': ['sum'],
        'CPUTIME_LEFT_SECONDS': ['sum'],
    }).sort_values(['STATE', ('CPUTIME_LEFT_SECONDS', 'sum')], ascending=False)

    if human_readable:
        aggregated_grouped['CPUTIME_LEFT_SECONDS', 'HR'] = aggregated_grouped['CPUTIME_LEFT_SECONDS', 'sum'].apply(
            seconds_to_hhmmss,
        )
        del aggregated_grouped[('CPUTIME_LEFT_SECONDS', 'sum')]
        aggregated_grouped.rename(
            columns={'CPUS': 'CPUs', 'CPUTIME_LEFT_SECONDS': 'CPU time remaining'},
            inplace=True,
        )
    else:
        aggregated_grouped.rename(
            columns={'CPUS': 'cpus_total', 'CPUTIME_LEFT_SECONDS': 'cputime_remaining_seconds_total'},
            inplace=True,
        )

    aggregated_grouped.columns = aggregated_grouped.columns.get_level_values(0)
    aggregated_grouped.index.names = ['State', 'Account']

    running_jobs = aggregated_grouped.loc["RUNNING"]
    pending_jobs = aggregated_grouped.loc["PENDING"]

    return running_jobs, pending_jobs


def create_job_detail_summary(job_detail: DataFrame, human_readable=True) -> DataFrame:
    def calc_cpu_time(row):
        cpu_max = row['TIME_SECONDS']*row['CPUS']
        total_used = row['AveCPU_SECONDS']*row['NODES']
        percentage_used = total_used / cpu_max if cpu_max != 0 else 0
        return f'{naturaldelta(timedelta(seconds=total_used))} / {naturaldelta(timedelta(seconds=cpu_max))} ({int(percentage_used*100)}%)'

    def calc_memory_used(row):
        def mem_str_to_bytes(mem_str):
            factor = {
                'B': 1,
                'K': 1024,
                'M': 1024**2,
                'G': 1024**3,
                'T': 1024**4,
                'P': 1024**5
            }
            try:
                return int(float(mem_str[:-1])*factor[mem_str[-1:]])
            except TypeError:
                return 0

        memory_allocated = mem_str_to_bytes(row['MIN_MEMORY']) * row['NODES']
        memory_used = mem_str_to_bytes(row['MaxRSS']) * row['NODES']
        percentage_used = memory_used / memory_allocated if memory_allocated != 0 else 0
        if human_readable:
            return f'{naturalsize(memory_used, binary=True)} / {naturalsize(memory_allocated, binary=True)} ({int(percentage_used*100)}%)'
        return f'{memory_used} / {memory_allocated} ({int(percentage_used*100)}%)'

    job_detail = job_detail.drop([
        'ACCOUNT',
        'JOBID.1',
        'TIME_LEFT',
        'NODELIST',
        'STATE',
        'UID',
        'AveCPU',
        'AveDiskRead',
        'AveDiskWrite',
    ], axis=1)

    job_detail['cpu_used'] = job_detail.apply(calc_cpu_time, axis=1)
    job_detail['memory_used'] = job_detail.apply(calc_memory_used, axis=1)

    job_detail = job_detail.drop([
        'TIME_SECONDS',
        'TIME_LEFT_SECONDS',
        'MaxRSS',
        'NTasks',
        'NODES',
        'AveCPU_SECONDS',
    ], axis=1)
    job_detail.set_index('JobID', inplace=True)
    return job_detail


def create_partition_memory_summary(human_readable: bool = True) -> dict:
    sinfo_mem = get_sinfo().groupby(['PARTITION']).agg({
        'FREE_MEM': ['sum'],
        'MEMORY': ['sum'],
    })
    sinfo_mem.columns = sinfo_mem.columns.get_level_values(0)

    sinfo_mem['FREE_MEM'] = sinfo_mem['FREE_MEM'].apply(lambda x: int(x * (1024 ** 2)))
    sinfo_mem['MEMORY'] = sinfo_mem['MEMORY'].apply(lambda x: int(x * (1024 ** 2)))
    sinfo_mem['FREE_MEM'] = sinfo_mem[['FREE_MEM', 'MEMORY']].min(axis=1)
    sinfo_mem['ALLOCATED_MEM'] = sinfo_mem['MEMORY'] - sinfo_mem['FREE_MEM']

    if human_readable:
        pass
    else:
        sinfo_mem.rename(
            columns={
                'MEMORY': 'total',
                'FREE_MEM': 'free',
                'ALLOCATED_MEM': 'allocated',
            },
            inplace=True,
        )
    return {'memory_total_bytes': sinfo_mem}


def create_partition_cpu_count_summary(human_readable: bool = True) -> dict:
    sinfo_cpu = get_sinfo()
    sinfo_cpu['allocated'] = sinfo_cpu['CPUS(A/I/O/T)'].apply(lambda x: int(x.split('/')[0]))
    sinfo_cpu['idle'] = sinfo_cpu['CPUS(A/I/O/T)'].apply(lambda x: int(x.split('/')[1]))
    sinfo_cpu['offline'] = sinfo_cpu['CPUS(A/I/O/T)'].apply(lambda x: int(x.split('/')[2]))
    sinfo_cpu['total'] = sinfo_cpu['CPUS(A/I/O/T)'].apply(lambda x: int(x.split('/')[3]))

    sinfo_cpu = sinfo_cpu.groupby(['PARTITION']).agg({
        'allocated': ['sum'],
        'idle': ['sum'],
        'offline': ['sum'],
        'total': ['sum'],
    })
    sinfo_cpu.columns = sinfo_cpu.columns.get_level_values(0)
    return {'cpu_state_count': sinfo_cpu}


def create_partition_cpu_load_summary(human_readable: bool = True) -> dict:
    sinfo_cpu = get_sinfo()
    sinfo_cpu["allocated"] = sinfo_cpu['CPUS(A/I/O/T)'].apply(lambda x: int(x.split('/')[0]))

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

