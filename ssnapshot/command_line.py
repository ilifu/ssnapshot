#!/usr/bin/env python3

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import OrderedDict
from datetime import datetime
from getpass import getuser
from json import dumps

from coloredlogs import install as coloredlogs_install

from ssnapshot.ssnapshot import (
    create_job_summaries,
    create_job_detail_summary,
    create_partition_summary,
    get_sinfo,
    get_squeue,
    get_sstat,
)


def create_arg_parser() -> ArgumentParser:
    new_parser = ArgumentParser(
        description='ssnapshot returns a brief summary of the status of slurm',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    new_parser.add_argument(
        '--verbose', '-v',
        default=0,
        action='count',
        help='0×v = ERRORs, 1×v = WARNINGs, 2×v = INFOs and 3×v = DEBUGs',
    )
    new_parser.add_argument(
        '--jobs', '-j',
        dest='tables',
        action='append_const',
        const='jobs',
        help='Show running / pending job summary information',
    )
    new_parser.add_argument(
        '--job-detail',
        nargs='?',
        metavar='user,...',
        default=False,
        const=getuser(),
        help=(
            'Show job details. Requires elevated privileges to get information from other users\' jobs. '
            'Use "ALL" to view all users jobs.'
        ),
    )
    new_parser.add_argument(
        '--partitions', '-p',
        dest='tables',
        action='append_const',
        const='partitions',
        help='Show partition summary information',
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

    new_parser.set_defaults(
        output='markdown',
        tables=[],
    )
    return new_parser


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

    output = OrderedDict([('header', {'value': 'Slurm Snapshot', 'time': str(datetime.now())})])

    if "jobs" in args.tables or args.job_detail:
        squeue = get_squeue()
        squeue.set_index('JOBID', inplace=True)
        if "jobs" in args.tables:
            running, pending = create_job_summaries(squeue)
            output['Running Jobs'] = {
                'type': 'table',
                'value': running,
            }
            output['Pending Jobs'] = {
                'type': 'table',
                'value': pending,
            }
        if args.job_detail:
            job_detail = get_sstat(squeue, args.job_detail.split(','))
            job_detail_summary = create_job_detail_summary(job_detail)
            output['Job Detail'] = {
                'type': 'table',
                'value': job_detail_summary,
            }
#            print(squeue.merge(job_detail, left_on='JOBID.1', right_on='JobID'))


    if "partitions" in args.tables:
        sinfo = get_sinfo()
        partitions = create_partition_summary(sinfo)
        output['Partition Summary'] = {
            'type': 'table',
            'value': partitions,
        }

    if args.output == 'markdown':
        header = output.get('header')
        if header:
            title = f'{header.get("value")}'
            time = header.get('time')
            if time:
                time = f' ({time})'
            print(f'# { title }{ time }')
        for name, value in output.items():
            output_type = value.get('type')
            if output_type == 'table':
                table_md = value.get('value').to_markdown()
                print(f'## {name}\n{table_md}\n\n')

    if args.output == 'json':
        for key, value in output.items():
            value_type = value.get('type')
            if value_type == 'table':
                value['value'] = value.get('value').to_dict()
        print(dumps(output, indent=2))

    if args.output == 'html':
        header = output.get('header')
        if header:
            title = f'{header.get("value")}'
            time = header.get('time')
            if time:
                time = f' ({time})'
            print(f'<h1>{title}{time}</h1>')
        for name, value in output.items():
            output_type = value.get('type')
            if output_type == 'table':
                table_html = value.get('value').to_html()
                print(f'<h2>{name}</h2>\n{table_html}\n')

if __name__ == '__main__':
    main()