#!/usr/bin/env python3

from argparse import ArgumentParser, FileType
from collections import OrderedDict
from datetime import datetime
import logging
from json import dumps
from sys import stdout
from time import sleep

from coloredlogs import install as coloredlogs_install

from ssnapshot.ssnapshot import (
    create_account_cpu_usage_summary,
    create_account_cputime_remaining_summary,
    create_fairshare_summaries,
    create_partition_memory_summary,
    create_partition_cpu_count_summary,
    create_partition_cpu_load_summary,
    sinfo_ttl_cache,
    squeue_ttl_cache,
    sstat_ttl_cache,
)


def create_arg_parser() -> ArgumentParser:
    new_parser = ArgumentParser(
        description='ssnapshot returns a brief summary of the status of slurm',
    )
    new_parser.add_argument(
        '--verbose', '-v',
        default=0,
        action='count',
        help='0×v = ERRORs, 1×v = WARNINGs, 2×v = INFOs and 3×v = DEBUGs',
    )
    new_parser.add_argument(
        '--daemonize', '-d',
        default=False,
        action='store_true',
        help='run in daemon mode',
    )
    new_parser.add_argument(
        '--sleep', '-s',
        default=300,
        type=int,
        help='Number of seconds to sleep between runs in daemon mode',
    )
    new_parser.add_argument(
        '--outfile', '-o',
        default=stdout,
        type=FileType('w'),
        help='Where to write output. Default is stdout',
    )

    new_parser.add_argument(
        '--accounts', '-a',
        dest='tables',
        action='append_const',
        const='accounts',
        help='Show account summary information. (Default: False)',
    )
    new_parser.add_argument(
        '--fairshare', '-f',
        dest='tables',
        action='append_const',
        const='fairshare',
        help='Show fairshare summary information. (Default: False)',
    )
    new_parser.add_argument(
        '--nodes', '-n',
        dest='tables',
        action='append_const',
        const='nodes',
        help='Show node summary information. (Default: False)',
    )
    new_parser.add_argument(
        '--partitions', '-p',
        dest='tables',
        action='append_const',
        const='partitions',
        help='Show partition summary information. (Default: False)',
    )

    output_group = new_parser.add_mutually_exclusive_group()
    output_group.add_argument(
        '--json',
        dest='output',
        action='store_const',
        const='json',
        help='Output is JSON',
    )
    output_group.add_argument(
        '--html',
        dest='output',
        action='store_const',
        const='html',
        help='Output is HTML',
    )
    output_group.add_argument(
        '--markdown',
        dest='output',
        action='store_const',
        const='markdown',
        help='Output is markdown',
    )
    output_group.add_argument(
        '--prometheus',
        dest='output',
        action='store_const',
        const='prometheus',
        help='Output is for prometheus exporter',
    )

    new_parser.set_defaults(
        output='markdown',
        tables=[],
        human_readable=True,
    )
    return new_parser


def generate_markdown(output: dict) -> str:
    lines = []
    header = output.get('header')
    if header:
        title = f'{header.get("value")}'
        time = header.get('time')
        if time:
            time = f' ({time})'
        lines.append(f'# {title}{time}')
    for name, value in output.items():
        output_type = value.get('type')
        if output_type == 'dataframe':
            table_md = value.get('dataframe').reset_index().to_markdown(index=False, floatfmt="0.4f")
            lines.append(f'## {name}\n{table_md}\n\n')
    return '\n'.join(lines)


def generate_html(output: dict) -> str:
    lines = []
    header = output.get('header')
    if header:
        title = f'{header.get("value")}'
        time = header.get('time')
        if time:
            time = f' ({time})'
        lines.append(f'<h1>{title}{time}</h1>')
    for name, value in output.items():
        output_type = value.get('type')
        if output_type == 'dataframe':
            table_html = value.get('dataframe').reset_index().to_html(index=False)
            lines.append(f'<h2>{name}</h2>\n{table_html}\n')
    return '\n'.join(lines)


def generate_json(output: dict) -> str:
    for key, value in output.items():
        value_type = value.get('type')
        if key == 'header':
            timestamp = value.get('time')
            if timestamp:
                output['header']['time'] = str(timestamp)
        if value_type == 'dataframe':
            value['dataframe'] = value.get('dataframe').reset_index().to_dict()
    return dumps(output, indent=2)


def generate_prometheus(output: dict) -> str:
    lines = []
    for key, value in output.items():
        output_type = value.get('type')
        if output_type == 'dataframe':
            table_name = key.lower().replace(' ', '_')
            dataframe = value.get('dataframe')
            index_names = [name.lower().replace(' ', '_') for name in dataframe.index.names]
            for row_index, row in dataframe.iterrows():
                if type(row_index) != tuple:
                    row_index = (row_index, )
                logging.debug(row_index)
                label_string = ", ".join([
                    f'{index_name}="{row_index[counter]}"' for counter, index_name in enumerate(index_names)
                ])
                logging.debug(label_string)
                for column_number, column in enumerate(dataframe.columns):
                    column_name = column.lower().replace(' ', '_').replace('/', 'per')
                    lines.append(
                        f'ssnapshot_{table_name}{{{label_string}, label="{column_name}"}} '
                        f'{row[column_number]:.6f}')
    return '\n'.join(lines) + '\n'


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    if args.verbose == 0:
        coloredlogs_install(level='ERROR')
    if args.verbose == 1:
        coloredlogs_install(level='WARNING')
    if args.verbose == 2:
        coloredlogs_install(level='INFO')
    if args.verbose >= 3:
        coloredlogs_install(level='DEBUG')

    output_method = {
        'html': generate_html,
        'json': generate_json,
        'markdown': generate_markdown,
        'prometheus': generate_prometheus,

    }.get(args.output)

    if args.output == 'prometheus':
        args.human_readable = False

    while True:
        for cache in sinfo_ttl_cache, squeue_ttl_cache, sstat_ttl_cache:
            cache.clear()
        output = OrderedDict([('header', {'value': 'Slurm Snapshot', 'time': datetime.now()})])

        summaries = []

        if "accounts" in args.tables:
            summaries.append(create_account_cpu_usage_summary())
            summaries.append(create_account_cputime_remaining_summary())
            # account_cpu_usage = create_account_cpu_usage_summary()
            # account_cputime_remaining = create_account_cputime_remaining_summary()
            # for info in [account_cpu_usage, account_cputime_remaining]:
            #     for table_name, data in info.items():
            #         output[table_name] = {
            #             'type': 'dataframe',
            #             'dataframe': data,
            #         }

        if "partitions" in args.tables:
            summaries.append(create_partition_memory_summary())
            summaries.append(create_partition_cpu_count_summary())
            summaries.append(create_partition_cpu_load_summary())
            # partition_mem = create_partition_memory_summary(args.human_readable)
            # partition_cpu = create_partition_cpu_count_summary(args.human_readable)
            # partition_load = create_partition_cpu_load_summary(args.human_readable)
            #
            # for info in [partition_mem, partition_cpu, partition_load]:
            #     for table_name, data in info.items():
            #         output[table_name] = {
            #             'type': 'dataframe',
            #             'dataframe': data,
            #         }

        if "fairshare" in args.tables:
            summaries.append(create_fairshare_summaries())
            # fairshare_account_summary = create_fairshare_summaries()
            #
            # for info in [fairshare_account_summary]:
            #     for table_name, data in info.items():
            #         output[table_name] = {
            #             'type': 'dataframe',
            #             'dataframe': data,
            #         }

        for summary in summaries:
            for table_name, data in summary.items():
                output[table_name] = {
                    'type': 'dataframe',
                    'dataframe': data,
                }

        output_string = ''

        if output_method:
            output_string = output_method(output)

        if output_string:
            try:
                args.outfile.truncate(0)
                args.outfile.seek(0, 0)
            except OSError:  # expected for stdout
                pass
            args.outfile.write(output_string)
            args.outfile.flush()

        if args.daemonize:
            sleep(args.sleep)
        else:
            break


if __name__ == '__main__':
    main()
