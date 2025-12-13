#!/usr/bin/env python3
"""
SimpleScalar 시뮬레이션 결과 샘플 데이터 생성기
테스트 및 데모용
"""
import os
import random

# 기본 템플릿
TEMPLATE = """sim-outorder: SimpleScalar/Alpha Tool Set version 3.0 of August, 2003.
Copyright (c) 1994-2003 by Todd M. Austin, Ph.D. and SimpleScalar, LLC.
All Rights Reserved. This version of SimpleScalar is licensed for academic
non-commercial use.

sim: command line: ./sim-outorder -config {config_name}.cfg {benchmark_cmd}

sim: simulation started @ Tue Nov 25 03:47:29 2025, options follow:

-fetch:ifqsize              {ifqsize}
-decode:width               {decode_width}
-issue:width                {issue_width}
-commit:width               {commit_width}
-bpred                  {bpred}
-cache:dl1       dl1:{dl1_nsets}:{dl1_bsize}:{dl1_assoc}:{dl1_repl}
-cache:dl2       ul2:1024:{dl2_bsize}:4:{dl2_repl}
-cache:il1       il1:512:{il1_bsize}:1:l
-res:ialu                   {ialu}
-res:imult                  {imult}
-res:fpalu                  {fpalu}
-res:fpmult                 {fpmult}
-ruu:size                  {ruu_size}
-lsq:size                   {lsq_size}

sim: ** simulation statistics **
sim_num_insn              {num_insn} # total number of instructions committed
sim_num_refs              {num_refs} # total number of loads and stores committed
sim_num_branches           {num_branches} # total number of branches committed
sim_elapsed_time                {elapsed_time} # total simulation time in seconds
sim_cycle                 {cycles} # total simulation time in cycles
sim_IPC                      {ipc:.4f} # instructions per cycle
sim_CPI                      {cpi:.4f} # cycles per instruction
bpred_{bpred}.lookups        {bpred_lookups} # total number of bpred lookups
bpred_{bpred}.updates        {bpred_updates} # total number of updates
bpred_{bpred}.addr_hits      {bpred_hits} # total number of address-predicted hits
bpred_{bpred}.misses          {bpred_misses} # total number of misses
bpred_{bpred}.bpred_dir_rate    {bpred_rate:.4f} # branch direction-prediction rate
dl1.accesses              {dl1_accesses} # total number of accesses
dl1.hits                  {dl1_hits} # total number of hits
dl1.misses                  {dl1_misses} # total number of misses
dl1.miss_rate                {dl1_miss_rate:.4f} # miss rate
ul2.accesses               {ul2_accesses} # total number of accesses
ul2.hits                   {ul2_hits} # total number of hits
ul2.misses                   {ul2_misses} # total number of misses
ul2.miss_rate                {ul2_miss_rate:.4f} # miss rate
il1.accesses              {il1_accesses} # total number of accesses
il1.hits                  {il1_hits} # total number of hits
il1.misses                 {il1_misses} # total number of misses
il1.miss_rate                {il1_miss_rate:.4f} # miss rate
"""

BENCHMARK_CMDS = {
    'gcc': 'benchmarks/cc1.alpha -O benchmarks/1stmt.i',
    'go': 'benchmarks/go.alpha 50 9 benchmarks/2stone9.in',
    'mcf': 'benchmarks/SPEC2000/spec2000binaries/mcf00.peak.ev6 benchmarks/SPEC2000/spec2000args/mcf/mcf.lgred.in',
    'gzip': 'benchmarks/SPEC2000/spec2000binaries/gzip00.peak.ev6 benchmarks/SPEC2000/spec2000args/gzip/gzip.lgred.graphic 1',
}

# 벤치마크별 기본 IPC (더 현실적인 값)
BASE_IPC = {
    'gcc': 0.85,
    'go': 0.72,
    'mcf': 0.35,
    'gzip': 0.95,
}

def generate_simulation_result(benchmark, config):
    """시뮬레이션 결과 생성"""
    
    # 기본 IPC 계산 (configuration에 따라 변화)
    base_ipc = BASE_IPC[benchmark]
    
    # IALU 영향 (더 많을수록 좋지만 diminishing returns)
    ialu = config.get('ialu', 4)
    ialu_factor = 0.7 + 0.15 * min(ialu, 4)
    
    # Issue width 영향
    width = config.get('issue_width', 4)
    width_factor = 0.6 + 0.2 * min(width, 4)
    
    # Branch predictor 영향
    bpred = config.get('bpred', 'bimod')
    bpred_factors = {
        'nottaken': 0.75,
        'taken': 0.80,
        '1bit': 0.88,
        '2bit': 0.92,
        '3bit': 0.95,
        'bimod': 0.93,
        '2lev': 0.96,
        'comb': 0.97,
    }
    bpred_factor = bpred_factors.get(bpred, 0.93)
    
    # Block size 영향 (공간 지역성)
    bsize = config.get('dl1_bsize', 32)
    bsize_factors = {16: 0.92, 32: 1.0, 64: 1.05, 128: 1.02}
    bsize_factor = bsize_factors.get(bsize, 1.0)
    
    # Associativity 영향
    assoc = config.get('dl1_assoc', 4)
    assoc_factors = {1: 0.85, 2: 0.95, 4: 1.0, 8: 1.02}
    assoc_factor = assoc_factors.get(assoc, 1.0)
    
    # Replacement policy 영향
    repl = config.get('dl1_repl', 'l')
    repl_factors = {'l': 1.0, 'f': 0.97, 'r': 0.94}
    repl_factor = repl_factors.get(repl, 1.0)
    
    # 최종 IPC 계산
    ipc = base_ipc * ialu_factor * width_factor * bpred_factor * bsize_factor * assoc_factor * repl_factor
    ipc += random.uniform(-0.02, 0.02)  # 약간의 랜덤 변동
    ipc = max(0.1, min(2.0, ipc))  # 범위 제한
    
    # 나머지 통계 생성
    num_insn = 337353533
    cycles = int(num_insn / ipc)
    cpi = 1 / ipc
    
    # Branch prediction 통계
    bpred_rate = bpred_factor - 0.05 + random.uniform(0, 0.1)
    bpred_lookups = int(num_insn * 0.22)
    bpred_updates = int(num_insn * 0.175)
    bpred_hits = int(bpred_updates * bpred_rate)
    bpred_misses = bpred_updates - bpred_hits
    
    # Cache 통계
    dl1_miss_rate = 0.02 + (0.03 if assoc == 1 else 0) + (0.01 if bsize < 32 else 0)
    dl1_accesses = int(num_insn * 0.37)
    dl1_hits = int(dl1_accesses * (1 - dl1_miss_rate))
    dl1_misses = dl1_accesses - dl1_hits
    
    ul2_miss_rate = 0.05 + random.uniform(0, 0.02)
    ul2_accesses = dl1_misses + int(num_insn * 0.035)
    ul2_hits = int(ul2_accesses * (1 - ul2_miss_rate))
    ul2_misses = ul2_accesses - ul2_hits
    
    il1_miss_rate = 0.025 + random.uniform(0, 0.01)
    il1_accesses = int(num_insn * 1.3)
    il1_hits = int(il1_accesses * (1 - il1_miss_rate))
    il1_misses = il1_accesses - il1_hits
    
    return TEMPLATE.format(
        config_name=f"T1_{benchmark}",
        benchmark_cmd=BENCHMARK_CMDS[benchmark],
        ifqsize=config.get('ifqsize', 4),
        decode_width=config.get('decode_width', 4),
        issue_width=width,
        commit_width=config.get('commit_width', 4),
        bpred=bpred,
        dl1_nsets=config.get('dl1_nsets', 128),
        dl1_bsize=bsize,
        dl1_assoc=assoc,
        dl1_repl=repl,
        dl2_bsize=config.get('dl2_bsize', 64),
        dl2_repl=config.get('dl2_repl', 'l'),
        il1_bsize=config.get('il1_bsize', 32),
        ialu=ialu,
        imult=config.get('imult', 1),
        fpalu=config.get('fpalu', 4),
        fpmult=config.get('fpmult', 1),
        ruu_size=config.get('ruu_size', 16),
        lsq_size=config.get('lsq_size', 8),
        num_insn=num_insn,
        num_refs=int(num_insn * 0.36),
        num_branches=int(num_insn * 0.175),
        elapsed_time=random.randint(200, 500),
        cycles=cycles,
        ipc=ipc,
        cpi=cpi,
        bpred_lookups=bpred_lookups,
        bpred_updates=bpred_updates,
        bpred_hits=bpred_hits,
        bpred_misses=bpred_misses,
        bpred_rate=bpred_rate,
        dl1_accesses=dl1_accesses,
        dl1_hits=dl1_hits,
        dl1_misses=dl1_misses,
        dl1_miss_rate=dl1_miss_rate,
        ul2_accesses=ul2_accesses,
        ul2_hits=ul2_hits,
        ul2_misses=ul2_misses,
        ul2_miss_rate=ul2_miss_rate,
        il1_accesses=il1_accesses,
        il1_hits=il1_hits,
        il1_misses=il1_misses,
        il1_miss_rate=il1_miss_rate,
    )

def main():
    os.makedirs('sample_results', exist_ok=True)
    
    benchmarks = ['gcc', 'go', 'mcf', 'gzip']
    
    # Section 3-1: ALU Units (1, 2, 4)
    for bench in benchmarks:
        for ialu in [1, 2, 4]:
            config = {'ialu': ialu, 'issue_width': 4, 'bpred': 'bimod', 
                     'dl1_bsize': 32, 'dl1_assoc': 4, 'dl1_repl': 'l'}
            result = generate_simulation_result(bench, config)
            filename = f"sample_results/result3_1_{ialu}{bench}.txt"
            with open(filename, 'w') as f:
                f.write(result)
            print(f"Generated: {filename}")
    
    # Section 3-2: Issue Width (1, 2, 4)
    for bench in benchmarks:
        for width in [1, 2, 4]:
            config = {'ialu': 4, 'issue_width': width, 'bpred': 'bimod',
                     'dl1_bsize': 32, 'dl1_assoc': 4, 'dl1_repl': 'l'}
            result = generate_simulation_result(bench, config)
            filename = f"sample_results/result3_2_width{width}_{bench}.txt"
            with open(filename, 'w') as f:
                f.write(result)
            print(f"Generated: {filename}")
    
    # Section 3-3: Branch Predictor
    for bench in benchmarks:
        for bpred in ['nottaken', 'taken', '1bit', '3bit']:
            config = {'ialu': 4, 'issue_width': 4, 'bpred': bpred,
                     'dl1_bsize': 32, 'dl1_assoc': 4, 'dl1_repl': 'l'}
            result = generate_simulation_result(bench, config)
            filename = f"sample_results/result3_3_{bpred}_{bench}.txt"
            with open(filename, 'w') as f:
                f.write(result)
            print(f"Generated: {filename}")
    
    # Section 4-1: Block Size (16, 32, 64, 128)
    for bench in benchmarks:
        for bsize in [16, 32, 64, 128]:
            config = {'ialu': 4, 'issue_width': 4, 'bpred': 'bimod',
                     'dl1_bsize': bsize, 'dl2_bsize': bsize, 'il1_bsize': bsize if bsize == 16 else 32,
                     'dl1_assoc': 4, 'dl1_repl': 'l'}
            result = generate_simulation_result(bench, config)
            filename = f"sample_results/result4_1_bsize{bsize}_{bench}.txt"
            with open(filename, 'w') as f:
                f.write(result)
            print(f"Generated: {filename}")
    
    # Section 4-2: Associativity (1, 2, 4)
    for bench in benchmarks:
        for assoc in [1, 2, 4]:
            config = {'ialu': 4, 'issue_width': 4, 'bpred': 'bimod',
                     'dl1_bsize': 32, 'dl1_assoc': assoc, 'dl1_repl': 'l'}
            result = generate_simulation_result(bench, config)
            filename = f"sample_results/result4_2_assoc{assoc}_{bench}.txt"
            with open(filename, 'w') as f:
                f.write(result)
            print(f"Generated: {filename}")
    
    # Section 4-3: Replacement Policy (LRU, FIFO, Random)
    for bench in benchmarks:
        for repl in ['l', 'f', 'r']:
            repl_name = {'l': 'lru', 'f': 'fifo', 'r': 'random'}[repl]
            config = {'ialu': 4, 'issue_width': 4, 'bpred': 'bimod',
                     'dl1_bsize': 64, 'dl1_assoc': 4, 'dl1_repl': repl, 'dl2_repl': repl}
            result = generate_simulation_result(bench, config)
            filename = f"sample_results/result4_3_{repl_name}_{bench}.txt"
            with open(filename, 'w') as f:
                f.write(result)
            print(f"Generated: {filename}")
    
    print(f"\n✅ Generated sample data in 'sample_results/' directory")

if __name__ == "__main__":
    main()
