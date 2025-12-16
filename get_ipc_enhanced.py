# -*- coding: utf-8 -*-
"""
================================================================================
SimpleScalar Simulation Result Analyzer
================================================================================
Computer Architecture Project - Architectural Simulation Analysis Tool
숭실대학교 전자정보공학부 IT융합전공 컴퓨터구조 팀 프로젝트용 자동화 분석 도구

Author: Seokjun RYU, Team 3
Date: 2025.12

이 스크립트는 SimpleScalar 시뮬레이션 결과 파일(.txt)을 파싱하여
프로젝트 요구사항에 맞는 그래프를 자동 생성합니다.

사용법:
    1. 시뮬레이션 결과 파일들을 현재 디렉토리에 저장
    2. 파일명 규칙: result{섹션}_{벤치마크}.txt 또는 자유 형식
    3. python get_ipc_enhanced.py 실행

생성되는 그래프:
    - Graph_3_1_ALU.png: 연산 장치 수에 따른 성능 분석
    - Graph_3_2_Width.png: Super-scalar width에 따른 성능 분석
    - Graph_3_3_Bpred.png: Branch predictor 종류에 따른 성능 분석
    - Graph_4_1_BlockSize.png: Cache block size에 따른 성능 분석
    - Graph_4_2_Assoc.png: Cache associativity에 따른 성능 분석
    - Graph_4_3_Replacement.png: Cache replacement policy에 따른 성능 분석
    - Graph_5_CacheOpt.png: 최적 Cache configuration 비교
    - Graph_6_SystemOpt.png: 최적 System configuration 비교
================================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import glob
import re
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 스타일 및 설정
# ==========================================

# 학술 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# 색상 팔레트 (전문적인 학술 스타일)
COLORS = {
    'primary': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
    'secondary': ['#3A7CA5', '#81B29A', '#F2CC8F', '#E07A5F'],
    'accent': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],
    'gradient': ['#1a535c', '#4ecdc4', '#f7fff7', '#ff6b6b', '#ffe66d'],
}

# 벤치마크 설정
BENCHMARKS = ['gcc', 'go', 'mcf', 'gzip']
BENCHMARK_DISPLAY = {'gcc': 'GCC', 'go': 'GO', 'mcf': 'MCF', 'gzip': 'GZIP'}

# 벤치마크 이름 매핑 (Command Line 내부 문자열 -> 표준 이름)
BENCHMARK_KEYWORDS = {
    'cc1': 'gcc',
    'gcc': 'gcc',
    'go': 'go',
    'mcf': 'mcf',
    'gzip': 'gzip'
}

# Branch Predictor 설명
BPRED_DESCRIPTIONS = {
    'nottaken': 'Not Taken\n(항상 분기 안함 예측)',
    'taken': 'Taken\n(항상 분기함 예측)',
    '1bit': '1-bit Counter\n(1비트 카운터)',
    '2bit': '2-bit Counter\n(2비트 포화 카운터)',
    '3bit': '3-bit Counter\n(3비트 포화 카운터)',
    'bimod': 'Bimodal\n(2비트 BHT)',
    '2lev': '2-Level\n(지역/전역 히스토리)',
    'comb': 'Combined\n(Tournament)',
}

# Replacement Policy 설명
REPL_DESCRIPTIONS = {
    'l': 'LRU\n(Least Recently Used)',
    'f': 'FIFO\n(First In First Out)',
    'r': 'Random\n(무작위 교체)',
}


# ==========================================
# 2. 파싱 함수
# ==========================================

def parse_cache_config(config_str):
    """
    Cache configuration 문자열 파싱
    형식: name:nsets:bsize:assoc:repl
    예: dl1:128:32:4:l
    """
    try:
        parts = config_str.split(':')
        if len(parts) >= 5:
            return {
                'name': parts[0],
                'nsets': int(parts[1]),
                'bsize': int(parts[2]),
                'assoc': int(parts[3]),
                'repl': parts[4],
                'size': int(parts[1]) * int(parts[2]) * int(parts[3])  # Total cache size
            }
    except:
        pass
    return None


def parse_filename_for_config(filename):
    """
    파일명에서 configuration 정보 추출
    예: result3_1_1gcc.txt -> section=3_1, ialu=1, benchmark=gcc
    예: result_gcc_ialu2_width4.txt -> benchmark=gcc, ialu=2, width=4
    """
    config = {}
    fname = filename.lower()
    
    # 섹션 정보 추출 (예: 3_1, 4_2 등)
    section_match = re.search(r'result?(\d+)[_-]?(\d+)?', fname)
    if section_match:
        config['section'] = section_match.group(1)
        if section_match.group(2):
            config['subsection'] = section_match.group(2)
    
    # IALU 수 추출 (파일명에서)
    ialu_patterns = [
        r'ialu[_-]?(\d+)',
        r'(\d+)ialu',
        r'alu[_-]?(\d+)',
        r'result\d+[_-]\d+[_-](\d+)',  # result3_1_1gcc.txt 형태
    ]
    for pattern in ialu_patterns:
        match = re.search(pattern, fname)
        if match:
            config['ialu_from_filename'] = int(match.group(1))
            break
    
    # Width 추출 (파일명에서)
    width_patterns = [
        r'width[_-]?(\d+)',
        r'(\d+)way',
        r'(\d+)[_-]?width',
    ]
    for pattern in width_patterns:
        match = re.search(pattern, fname)
        if match:
            config['width_from_filename'] = int(match.group(1))
            break
    
    # Block size 추출 (파일명에서)
    bsize_patterns = [
        r'bsize[_-]?(\d+)',
        r'block[_-]?(\d+)',
        r'(\d+)b(?:yte)?',
    ]
    for pattern in bsize_patterns:
        match = re.search(pattern, fname)
        if match:
            val = int(match.group(1))
            if val in [16, 32, 64, 128, 256]:
                config['bsize_from_filename'] = val
                break
    
    # Associativity 추출 (파일명에서)
    assoc_patterns = [
        r'assoc[_-]?(\d+)',
        r'(\d+)[_-]?assoc',
        r'(\d+)way(?!width)',
    ]
    for pattern in assoc_patterns:
        match = re.search(pattern, fname)
        if match:
            config['assoc_from_filename'] = int(match.group(1))
            break
    
    # Branch predictor 추출 (파일명에서)
    bpred_keywords = ['nottaken', 'taken', '1bit', '2bit', '3bit', 'bimod', '2lev', 'comb']
    for bp in bpred_keywords:
        if bp in fname:
            config['bpred_from_filename'] = bp
            break
    
    # Replacement policy 추출 (파일명에서)
    repl_patterns = {
        'lru': 'l',
        'fifo': 'f',
        'random': 'r',
        '_l_': 'l',
        '_f_': 'f',
        '_r_': 'r',
    }
    for pattern, repl in repl_patterns.items():
        if pattern in fname:
            config['repl_from_filename'] = repl
            break
    
    return config


def parse_simulation_file(filepath):
    """
    단일 시뮬레이션 결과 파일 파싱
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        filename = os.path.basename(filepath)
        result = {
            'filename': filename,
            'filepath': filepath,
        }
        
        # 파일명에서 추가 정보 추출
        filename_config = parse_filename_for_config(filename)
        result.update(filename_config)
        
        # 1. 벤치마크 이름 파싱
        cmd_match = re.search(r'sim: command line:(.*)', content)
        if cmd_match:
            cmd_line = cmd_match.group(1).lower()
            for key, val in BENCHMARK_KEYWORDS.items():
                if key in cmd_line:
                    result['benchmark'] = val
                    break
        
        if 'benchmark' not in result:
            # 파일명에서 벤치마크 추출 시도
            fname_lower = filepath.lower()
            for key, val in BENCHMARK_KEYWORDS.items():
                if key in fname_lower:
                    result['benchmark'] = val
                    break
        
        if 'benchmark' not in result:
            return None
        
        # 2. 성능 메트릭 파싱
        metrics = {
            'sim_IPC': r'sim_IPC\s+([\d.]+)',
            'sim_CPI': r'sim_CPI\s+([\d.]+)',
            'sim_cycle': r'sim_cycle\s+(\d+)',
            'sim_num_insn': r'sim_num_insn\s+(\d+)',
            'sim_num_branches': r'sim_num_branches\s+(\d+)',
        }
        
        for key, pattern in metrics.items():
            match = re.search(pattern, content)
            if match:
                result[key] = float(match.group(1))
        
        if 'sim_IPC' not in result:
            return None
        
        # 3. Processor Configuration 파싱
        proc_configs = {
            'fetch_ifqsize': (r'-fetch:ifqsize\s+(\d+)', 4),
            'decode_width': (r'-decode:width\s+(\d+)', 4),
            'issue_width': (r'-issue:width\s+(\d+)', 4),
            'commit_width': (r'-commit:width\s+(\d+)', 4),
            'res_ialu': (r'-res:ialu\s+(\d+)', 4),
            'res_imult': (r'-res:imult\s+(\d+)', 1),
            'res_fpalu': (r'-res:fpalu\s+(\d+)', 4),
            'res_fpmult': (r'-res:fpmult\s+(\d+)', 1),
            'ruu_size': (r'-ruu:size\s+(\d+)', 16),
            'lsq_size': (r'-lsq:size\s+(\d+)', 8),
        }
        
        for key, (pattern, default) in proc_configs.items():
            match = re.search(pattern, content)
            result[key] = int(match.group(1)) if match else default
        
        # 4. Branch Predictor 파싱
        bpred_match = re.search(r'-bpred\s+(\w+)', content)
        result['bpred'] = bpred_match.group(1) if bpred_match else 'bimod'
        
        # Branch prediction accuracy 파싱
        bpred_patterns = {
            'bpred_lookups': r'bpred_\w+\.lookups\s+(\d+)',
            'bpred_updates': r'bpred_\w+\.updates\s+(\d+)',
            'bpred_hits': r'bpred_\w+\.addr_hits\s+(\d+)',
            'bpred_misses': r'bpred_\w+\.misses\s+(\d+)',
            'bpred_dir_rate': r'bpred_\w+\.bpred_dir_rate\s+([\d.]+)',
        }
        
        for key, pattern in bpred_patterns.items():
            match = re.search(pattern, content)
            if match:
                result[key] = float(match.group(1))
        
        # 5. Cache Configuration 파싱
        cache_configs = ['dl1', 'dl2', 'il1', 'il2']
        for cache in cache_configs:
            pattern = rf'-cache:{cache}\s+(\S+)'
            match = re.search(pattern, content)
            if match:
                config_str = match.group(1)
                if config_str not in ['none', 'dl1', 'dl2']:
                    parsed = parse_cache_config(config_str)
                    if parsed:
                        result[f'{cache}_nsets'] = parsed['nsets']
                        result[f'{cache}_bsize'] = parsed['bsize']
                        result[f'{cache}_assoc'] = parsed['assoc']
                        result[f'{cache}_repl'] = parsed['repl']
                        result[f'{cache}_size'] = parsed['size']
        
        # 6. Cache 성능 메트릭 파싱
        cache_metrics = {
            'dl1_miss_rate': r'dl1\.miss_rate\s+([\d.]+)',
            'dl1_hits': r'dl1\.hits\s+(\d+)',
            'dl1_misses': r'dl1\.misses\s+(\d+)',
            'dl1_accesses': r'dl1\.accesses\s+(\d+)',
            'ul2_miss_rate': r'ul2\.miss_rate\s+([\d.]+)',
            'ul2_hits': r'ul2\.hits\s+(\d+)',
            'ul2_misses': r'ul2\.misses\s+(\d+)',
            'il1_miss_rate': r'il1\.miss_rate\s+([\d.]+)',
        }
        
        for key, pattern in cache_metrics.items():
            match = re.search(pattern, content)
            if match:
                result[key] = float(match.group(1))
        
        return result
        
    except Exception as e:
        print(f"⚠️ Error parsing {filepath}: {e}")
        return None


def parse_all_simulation_files(directory='.'):
    """
    디렉토리 내 모든 시뮬레이션 결과 파일 파싱
    """
    file_patterns = ['*.txt', 'result*.txt', '*result*.txt']
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    files = list(set(files))  # 중복 제거
    
    print("=" * 80)
    print("📂 SimpleScalar Simulation Result Parser")
    print("=" * 80)
    print(f"\n📁 Found {len(files)} potential result files\n")
    
    data = []
    
    print(f"{'Filename':<35} {'Bench':<6} {'IPC':<8} {'IALU':<5} {'Width':<6} {'Bpred':<10} {'DL1 Config'}")
    print("-" * 100)
    
    for filepath in sorted(files):
        result = parse_simulation_file(filepath)
        if result:
            data.append(result)
            
            # 파싱 결과 출력
            dl1_config = f"{result.get('dl1_bsize', '-')}/{result.get('dl1_assoc', '-')}/{result.get('dl1_repl', '-')}"
            print(f"{result['filename'][:35]:<35} {result['benchmark']:<6} "
                  f"{result['sim_IPC']:<8.4f} {result.get('res_ialu', '-'):<5} "
                  f"{result.get('issue_width', '-'):<6} {result.get('bpred', '-'):<10} {dl1_config}")
    
    print("-" * 100)
    print(f"✅ Successfully parsed {len(data)} files\n")
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # 벤치마크 정렬
    df['benchmark'] = pd.Categorical(df['benchmark'], categories=BENCHMARKS, ordered=True)
    df = df.sort_values('benchmark')
    
    return df


# ==========================================
# 3. 그래프 생성 함수
# ==========================================

def create_grouped_bar_chart(df, group_col, group_values, title, xlabel, ylabel, 
                              filename, label_map=None, show_improvement=False,
                              baseline_value=None, figsize=(12, 7)):
    """
    그룹화된 막대 그래프 생성 (전문적인 스타일)
    """
    if df.empty:
        print(f"⚠️ Skipping {filename}: No data available")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(BENCHMARKS))
    n_groups = len(group_values)
    width = 0.7 / n_groups
    
    colors = COLORS['accent'][:n_groups]
    
    bars_data = {}
    max_ipc = 0
    
    for i, val in enumerate(group_values):
        ipcs = []
        for bench in BENCHMARKS:
            bench_df = df[df['benchmark'] == bench]
            if group_col in bench_df.columns:
                if isinstance(val, (int, float)):
                    row = bench_df[bench_df[group_col] == val]
                else:
                    row = bench_df[bench_df[group_col] == str(val)]
                
                if not row.empty:
                    ipc = row['sim_IPC'].max()
                    ipcs.append(ipc)
                    max_ipc = max(max_ipc, ipc)
                else:
                    ipcs.append(0)
            else:
                ipcs.append(0)
        
        bars_data[val] = ipcs
        
        label = str(val)
        if label_map and val in label_map:
            label = label_map[val]
        
        offset = (i - n_groups/2 + 0.5) * width
        bars = ax.bar(x + offset, ipcs, width, label=label, color=colors[i % len(colors)],
                     edgecolor='white', linewidth=0.5, alpha=0.9)
        
        # 값 표시
        for bar, ipc in zip(bars, ipcs):
            if ipc > 0:
                height = bar.get_height()
                ax.annotate(f'{ipc:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8, fontweight='bold')
    
    # 축 설정
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=12)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_DISPLAY[b] for b in BENCHMARKS], fontsize=11, fontweight='bold')
    ax.set_ylim(0, max_ipc * 1.2 if max_ipc > 0 else 1)
    
    # 범례
    ax.legend(title=xlabel, loc='upper left', framealpha=0.9, edgecolor='gray')
    
    # 그리드
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # 스타일 개선
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()
    
    return bars_data


def create_line_chart(df, x_col, x_values, title, xlabel, ylabel, filename, figsize=(14, 10)):
    """
    벤치마크별 라인 차트 (2x2 서브플롯)
    """
    if df.empty:
        print(f"⚠️ Skipping {filename}: No data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    colors = COLORS['accent']
    markers = ['o', 's', '^', 'D']
    
    for idx, bench in enumerate(BENCHMARKS):
        ax = axes[idx]
        bench_df = df[df['benchmark'] == bench]
        
        ipcs = []
        valid_x = []
        
        for x_val in x_values:
            row = bench_df[bench_df[x_col] == x_val]
            if not row.empty:
                ipcs.append(row['sim_IPC'].max())
                valid_x.append(x_val)
            else:
                ipcs.append(np.nan)
                valid_x.append(x_val)
        
        # NaN이 아닌 값만 플롯
        valid_indices = [i for i, v in enumerate(ipcs) if not np.isnan(v)]
        plot_x = [valid_x[i] for i in valid_indices]
        plot_y = [ipcs[i] for i in valid_indices]
        
        if plot_y:
            ax.plot(plot_x, plot_y, marker=markers[idx], markersize=10, 
                   linewidth=2.5, color=colors[idx], label=BENCHMARK_DISPLAY[bench])
            
            # 최대값 표시
            max_ipc = max(plot_y)
            max_idx = plot_y.index(max_ipc)
            ax.annotate(f'Best: {max_ipc:.4f}',
                       xy=(plot_x[max_idx], max_ipc),
                       xytext=(10, 10),
                       textcoords='offset points',
                       fontsize=10, fontweight='bold', color='red',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
            
            # 각 포인트에 값 표시
            for x, y in zip(plot_x, plot_y):
                ax.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, -15),
                           textcoords='offset points', ha='center', fontsize=9)
        
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'{BENCHMARK_DISPLAY[bench]}', fontweight='bold', fontsize=13)
        ax.set_xticks(x_values)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if plot_y:
            y_min, y_max = min(plot_y), max(plot_y)
            margin = (y_max - y_min) * 0.15 if y_max != y_min else 0.1
            ax.set_ylim(y_min - margin, y_max + margin)
    
    fig.suptitle(title, fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


def create_comparison_chart(df, configs, title, filename, figsize=(14, 8)):
    """
    Baseline vs Optimized 비교 차트
    """
    if df.empty:
        print(f"⚠️ Skipping {filename}: No data available")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(BENCHMARKS))
    width = 0.35
    
    baseline_ipcs = []
    optimized_ipcs = []
    improvements = []
    
    for bench in BENCHMARKS:
        bench_df = df[df['benchmark'] == bench]
        if not bench_df.empty:
            baseline = bench_df['sim_IPC'].min()
            optimized = bench_df['sim_IPC'].max()
            baseline_ipcs.append(baseline)
            optimized_ipcs.append(optimized)
            if baseline > 0:
                improvements.append((optimized - baseline) / baseline * 100)
            else:
                improvements.append(0)
        else:
            baseline_ipcs.append(0)
            optimized_ipcs.append(0)
            improvements.append(0)
    
    # 막대 그래프
    bars1 = ax.bar(x - width/2, baseline_ipcs, width, label='Baseline', 
                   color=COLORS['secondary'][3], edgecolor='white', alpha=0.9)
    bars2 = ax.bar(x + width/2, optimized_ipcs, width, label='Optimized',
                   color=COLORS['accent'][1], edgecolor='white', alpha=0.9)
    
    # 개선율 표시
    for i, (bar, imp) in enumerate(zip(bars2, improvements)):
        if imp > 0:
            ax.annotate(f'+{imp:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 5),
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold', color='#C73E1D')
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height/2),
                           ha='center', va='center',
                           fontsize=9, fontweight='bold', color='white')
    
    ax.set_xlabel('Benchmark', fontweight='bold', fontsize=12)
    ax.set_ylabel('IPC (Instructions Per Cycle)', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_DISPLAY[b] for b in BENCHMARKS], fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 평균 개선율 표시
    avg_improvement = np.mean([i for i in improvements if i > 0])
    ax.text(0.02, 0.98, f'Average Improvement: +{avg_improvement:.2f}%',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


def create_heatmap(df, x_col, y_col, x_values, y_values, title, filename,
                   x_label_map=None, y_label_map=None, figsize=(14, 10)):
    """
    벤치마크별 히트맵 (2x2 서브플롯)
    """
    if df.empty:
        print(f"⚠️ Skipping {filename}: No data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, bench in enumerate(BENCHMARKS):
        ax = axes[idx]
        bench_df = df[df['benchmark'] == bench]
        
        # 히트맵 데이터 구성
        heatmap_data = np.zeros((len(y_values), len(x_values)))
        
        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                row = bench_df[(bench_df[x_col] == x_val) & (bench_df[y_col] == y_val)]
                if not row.empty:
                    heatmap_data[i, j] = row['sim_IPC'].max()
        
        # 히트맵 그리기
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        
        # 축 설정
        x_labels = [x_label_map.get(v, str(v)) if x_label_map else str(v) for v in x_values]
        y_labels = [y_label_map.get(v, str(v)) if y_label_map else str(v) for v in y_values]
        
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        
        # 값 표시
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                if heatmap_data[i, j] > 0:
                    text_color = 'white' if heatmap_data[i, j] > heatmap_data.max() * 0.7 else 'black'
                    ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                           ha='center', va='center', color=text_color, fontweight='bold')
        
        ax.set_title(f'{BENCHMARK_DISPLAY[bench]}', fontweight='bold', fontsize=13)
        
        # 컬러바
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('IPC', fontweight='bold')
    
    fig.suptitle(title, fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


def create_radar_chart(df, metrics, title, filename, figsize=(12, 10)):
    """
    벤치마크별 레이더 차트
    """
    if df.empty:
        print(f"⚠️ Skipping {filename}: No data available")
        return
    
    # 각 벤치마크별 메트릭 수집
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 닫힌 다각형
    
    colors = COLORS['accent']
    
    for idx, bench in enumerate(BENCHMARKS):
        bench_df = df[df['benchmark'] == bench]
        if bench_df.empty:
            continue
        
        values = []
        for metric in metrics:
            if metric in bench_df.columns:
                val = bench_df[metric].max()
                values.append(val if not np.isnan(val) else 0)
            else:
                values.append(0)
        
        # 정규화
        max_vals = [df[m].max() if m in df.columns else 1 for m in metrics]
        norm_values = [v / mv if mv > 0 else 0 for v, mv in zip(values, max_vals)]
        norm_values += norm_values[:1]
        
        ax.plot(angles, norm_values, 'o-', linewidth=2, label=BENCHMARK_DISPLAY[bench],
               color=colors[idx % len(colors)])
        ax.fill(angles, norm_values, alpha=0.25, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


# ==========================================
# 4. 섹션별 그래프 생성 함수
# ==========================================

def run_section_3_1(df):
    """
    ③-1 연산 장치 수에 따른 성능 분석
    Configuration: res:ialu, res:imult, res:fpalu, res:fpmult
    """
    print("\n📊 Section 3-1: Functional Units Analysis")
    
    # res_ialu 값이 있는 데이터만 필터링
    alu_df = df[df['res_ialu'].notna()].copy()
    
    if alu_df.empty:
        print("⚠️ No Functional Units configuration data found")
        return
    
    alu_values = sorted(alu_df['res_ialu'].unique())
    target_values = [v for v in [1, 2, 4] if v in alu_values]
    
    if not target_values:
        target_values = alu_values[:3]
    
    create_grouped_bar_chart(
        alu_df, 'res_ialu', target_values,
        title='3-1. IPC vs Number of Functional Units',
        xlabel='Number of Functional Units',
        ylabel='IPC (Instructions Per Cycle)',
        filename='Graph_3_1_FU.png',
        label_map={1: '1 Unit', 2: '2 Units', 4: '4 Units'}
    )


def run_section_3_2(df):
    """
    ③-2 Super-scalar width에 따른 성능 분석
    Configuration: issue:width
    """
    print("\n📊 Section 3-2: Issue Width Analysis")
    
    width_df = df[df['issue_width'].notna()].copy()
    
    if width_df.empty:
        print("⚠️ No Issue Width configuration data found")
        return
    
    width_values = sorted(width_df['issue_width'].unique())
    target_values = [v for v in [1, 2, 4] if v in width_values]
    
    if not target_values:
        target_values = width_values[:3]
    
    create_grouped_bar_chart(
        width_df, 'issue_width', target_values,
        title='3-2. IPC vs Super-scalar Issue Width',
        xlabel='Issue Width (Instructions/Cycle)',
        ylabel='IPC (Instructions Per Cycle)',
        filename='Graph_3_2_Width.png',
        label_map={1: '1-way (Scalar)', 2: '2-way', 4: '4-way'}
    )


def run_section_3_3(df):
    """
    ③-3 Branch Predictor 종류에 따른 성능 분석
    Configuration: bpred
    """
    print("\n📊 Section 3-3: Branch Predictor Analysis")
    
    bpred_df = df[df['bpred'].notna()].copy()
    
    if bpred_df.empty:
        print("⚠️ No Branch Predictor configuration data found")
        return
    
    all_bpreds = bpred_df['bpred'].unique().tolist()
    
    # 프로젝트 요구사항: nottaken, taken, 1bit, 3bit
    priority_list = ['nottaken', 'taken', '1bit', '3bit', '2bit', '2lev', 'comb']
    
    targets = [p for p in priority_list if p in all_bpreds]
    
    # 나머지 추가 (bimod 제외)
    for b in all_bpreds:
        if b not in targets and b != 'bimod':
            targets.append(b)
    
    if not targets:
        print("⚠️ No valid Branch Predictor data (excluding bimod)")
        return
    
    create_grouped_bar_chart(
        bpred_df, 'bpred', targets[:5],  # 최대 5개
        title='3-3. IPC vs Branch Predictor Type',
        xlabel='Branch Predictor',
        ylabel='IPC (Instructions Per Cycle)',
        filename='Graph_3_3_Bpred.png',
        label_map={k: BPRED_DESCRIPTIONS.get(k, k).split('\n')[0] for k in targets}
    )


def run_section_4_1(df):
    """
    ④-1 Cache Block Size에 따른 성능 분석
    Configuration: cache:dl1, cache:dl2 block size
    """
    print("\n📊 Section 4-1: Cache Block Size Analysis")
    
    bsize_df = df[df['dl1_bsize'].notna()].copy()
    
    if bsize_df.empty:
        print("⚠️ No Cache Block Size configuration data found")
        return
    
    bsize_values = sorted(bsize_df['dl1_bsize'].unique())
    target_values = [v for v in [16, 32, 64, 128] if v in bsize_values]
    
    if not target_values:
        target_values = bsize_values[:4]
    
    create_line_chart(
        bsize_df, 'dl1_bsize', target_values,
        title='4-1. IPC vs Cache Block Size',
        xlabel='Block Size (Bytes)',
        ylabel='IPC (Instructions Per Cycle)',
        filename='Graph_4_1_BlockSize.png'
    )


def run_section_4_2(df):
    """
    ④-2 Cache Associativity에 따른 성능 분석
    Configuration: cache:dl1 associativity
    """
    print("\n📊 Section 4-2: Cache Associativity Analysis")
    
    assoc_df = df[df['dl1_assoc'].notna()].copy()
    
    if assoc_df.empty:
        print("⚠️ No Cache Associativity configuration data found")
        return
    
    assoc_values = sorted(assoc_df['dl1_assoc'].unique())
    target_values = [v for v in [1, 2, 4] if v in assoc_values]
    
    if not target_values:
        target_values = assoc_values[:3]
    
    create_grouped_bar_chart(
        assoc_df, 'dl1_assoc', target_values,
        title='4-2. IPC vs Cache Associativity (L1 Data Cache)',
        xlabel='Associativity',
        ylabel='IPC (Instructions Per Cycle)',
        filename='Graph_4_2_Assoc.png',
        label_map={1: 'Direct-mapped', 2: '2-way', 4: '4-way', 8: '8-way'}
    )


def run_section_4_3(df):
    """
    ④-3 Cache Replacement Policy에 따른 성능 분석
    Configuration: cache:dl1 replacement policy
    """
    print("\n📊 Section 4-3: Cache Replacement Policy Analysis")
    
    repl_df = df[df['dl1_repl'].notna()].copy()
    
    if repl_df.empty:
        print("⚠️ No Cache Replacement Policy configuration data found")
        return
    
    repl_values = repl_df['dl1_repl'].unique().tolist()
    target_values = [v for v in ['l', 'f', 'r'] if v in repl_values]
    
    if not target_values:
        target_values = repl_values[:3]
    
    create_grouped_bar_chart(
        repl_df, 'dl1_repl', target_values,
        title='4-3. IPC vs Cache Replacement Policy (Block Size: 64B)',
        xlabel='Replacement Policy',
        ylabel='IPC (Instructions Per Cycle)',
        filename='Graph_4_3_Replacement.png',
        label_map={'l': 'LRU', 'f': 'FIFO', 'r': 'Random'}
    )


def create_full_heatmap(df, bsize_values, assoc_values, repl_values, filename):
    """
    4개 Application × 3개 Replacement Policy = 12개 서브플롯 히트맵
    행: Application (GCC, GO, MCF, GZIP)
    열: Replacement Policy (LRU, FIFO, Random)
    """
    fig, axes = plt.subplots(4, 3, figsize=(14, 16))
    
    repl_names = {'l': 'LRU', 'f': 'FIFO', 'r': 'Random'}
    
    # 전체 데이터에서 Application별 최대/최소값 찾기 (각 행별로 컬러스케일 통일)
    bench_ranges = {}
    for bench in BENCHMARKS:
        bench_df = df[df['benchmark'] == bench]
        all_ipcs = bench_df['sim_IPC'].dropna().tolist()
        if all_ipcs:
            bench_ranges[bench] = (min(all_ipcs), max(all_ipcs))
        else:
            bench_ranges[bench] = (0, 1)
    
    best_configs = {}
    
    for row_idx, bench in enumerate(BENCHMARKS):
        bench_df = df[df['benchmark'] == bench]
        vmin, vmax = bench_ranges[bench]
        
        best_config = {'ipc': 0, 'bsize': 0, 'assoc': 0, 'repl': ''}
        
        for col_idx, repl in enumerate(repl_values):
            ax = axes[row_idx, col_idx]
            
            # 히트맵 데이터 구성
            heatmap_data = np.full((len(assoc_values), len(bsize_values)), np.nan)
            
            for i, assoc in enumerate(assoc_values):
                for j, bsize in enumerate(bsize_values):
                    row = bench_df[(bench_df['dl1_bsize'] == bsize) & 
                                  (bench_df['dl1_assoc'] == assoc) & 
                                  (bench_df['dl1_repl'] == repl)]
                    if not row.empty:
                        ipc = row['sim_IPC'].max()
                        heatmap_data[i, j] = ipc
                        if ipc > best_config['ipc']:
                            best_config = {'ipc': ipc, 'bsize': bsize, 'assoc': assoc, 'repl': repl}
            
            # 히트맵 그리기
            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)
            
            # 축 설정
            ax.set_xticks(np.arange(len(bsize_values)))
            ax.set_yticks(np.arange(len(assoc_values)))
            ax.set_xticklabels([f'{v}B' for v in bsize_values], fontsize=9)
            ax.set_yticklabels([f'{v}-way' for v in assoc_values], fontsize=9)
            
            # 값 표시
            for i in range(len(assoc_values)):
                for j in range(len(bsize_values)):
                    if not np.isnan(heatmap_data[i, j]):
                        text_color = 'white' if heatmap_data[i, j] > (vmin + vmax) / 2 else 'black'
                        ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                               ha='center', va='center', color=text_color, 
                               fontweight='bold', fontsize=8)
            
            # 첫 번째 행에만 Replacement Policy 제목
            if row_idx == 0:
                ax.set_title(f'{repl_names[repl]}', fontweight='bold', fontsize=12)
            
            # 첫 번째 열에만 Application 이름 (Y축 라벨)
            if col_idx == 0:
                ax.set_ylabel(f'{BENCHMARK_DISPLAY[bench]}', fontweight='bold', fontsize=11)
            
            # 마지막 행에만 X축 라벨
            if row_idx == len(BENCHMARKS) - 1:
                ax.set_xlabel('Block Size', fontsize=10)
        
        # 각 Application의 최적 config 저장
        if best_config['ipc'] > 0:
            best_configs[bench] = best_config
        
        # 각 행 끝에 컬러바 추가 (오른쪽으로 더 이동)
        cbar = fig.colorbar(im, ax=axes[row_idx, :], fraction=0.02, pad=0.05)
        cbar.set_label('IPC', fontsize=9)
    
    fig.suptitle('5-1. IPC Heatmap: Block Size × Associativity × Replacement Policy',
                 fontweight='bold', fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()
    
    return best_configs


def create_summary_table(best_configs, filename):
    """
    모든 Application의 최적 configuration 요약 테이블 생성
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    repl_names = {'l': 'LRU', 'f': 'FIFO', 'r': 'Random'}
    
    # 테이블 데이터 구성
    table_data = []
    for bench in BENCHMARKS:
        if bench in best_configs and best_configs[bench]:
            cfg = best_configs[bench]
            table_data.append([
                BENCHMARK_DISPLAY[bench],
                f"{cfg['bsize']}B",
                f"{cfg['assoc']}-way",
                repl_names.get(cfg['repl'], cfg['repl']),
                f"{cfg['ipc']:.4f}"
            ])
        else:
            table_data.append([BENCHMARK_DISPLAY[bench], '-', '-', '-', '-'])
    
    # 테이블 생성
    columns = ['Application', 'Block Size', 'Associativity', 'Replacement', 'Best IPC']
    
    table = ax.table(cellText=table_data,
                     colLabels=columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.18, 0.18, 0.18, 0.18, 0.18])
    
    # 스타일 설정
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # 헤더 스타일
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 데이터 행 스타일
    colors = ['#f0f0f0', '#ffffff']
    for i in range(len(table_data)):
        for j in range(len(columns)):
            table[(i+1, j)].set_facecolor(colors[i % 2])
            if j == 4:  # IPC 열 강조
                table[(i+1, j)].set_text_props(fontweight='bold', color='#C73E1D')
    
    ax.set_title('5-2. Optimal Cache Configuration Summary\n(Block Size × Associativity × Replacement Policy)',
                 fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


def create_summary_bar_chart(best_configs, filename):
    """
    최적 configuration 비교 막대 그래프
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    repl_names = {'l': 'LRU', 'f': 'FIFO', 'r': 'Random'}
    
    benchmarks = []
    ipcs = []
    labels = []
    
    for bench in BENCHMARKS:
        if bench in best_configs and best_configs[bench]:
            cfg = best_configs[bench]
            benchmarks.append(BENCHMARK_DISPLAY[bench])
            ipcs.append(cfg['ipc'])
            labels.append(f"{cfg['bsize']}B\n{cfg['assoc']}-way\n{repl_names.get(cfg['repl'], cfg['repl'])}")
    
    if not benchmarks:
        print("⚠️ No data for summary bar chart")
        plt.close()
        return
    
    x = np.arange(len(benchmarks))
    bars = ax.bar(x, ipcs, color=COLORS['accent'][:len(benchmarks)], 
                  edgecolor='white', linewidth=2, alpha=0.9)
    
    # 값과 configuration 표시
    for i, (bar, ipc, label) in enumerate(zip(bars, ipcs, labels)):
        # IPC 값
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ipc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        # Configuration
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                label, ha='center', va='center', fontweight='bold', 
                fontsize=9, color='white')
    
    ax.set_xlabel('Benchmark', fontweight='bold', fontsize=12)
    ax.set_ylabel('IPC (Instructions Per Cycle)', fontweight='bold', fontsize=12)
    ax.set_title('5-3. Optimal Cache Configuration Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontweight='bold')
    ax.set_ylim(0, max(ipcs) * 1.2 if ipcs else 1)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


def run_section_5(df):
    """
    ⑤ 최적 Cache Configuration 탐색
    Block Size × Associativity × Replacement Policy 조합 분석
    """
    print("\n📊 Section 5: Optimal Cache Configuration")
    
    # 캐시 관련 데이터만 필터링
    cache_df = df[df['dl1_bsize'].notna() & df['dl1_assoc'].notna() & df['dl1_repl'].notna()].copy()
    
    if cache_df.empty:
        print("⚠️ No complete Cache configuration data found")
        return
    
    # 사용 가능한 값들 추출
    bsize_values = sorted(cache_df['dl1_bsize'].dropna().unique())
    assoc_values = sorted(cache_df['dl1_assoc'].dropna().unique())
    repl_values = [r for r in ['l', 'f', 'r'] if r in cache_df['dl1_repl'].unique()]
    
    # 타겟 값들 (프로젝트 요구사항)
    target_bsize = [v for v in [16, 32, 64, 128] if v in bsize_values]
    target_assoc = [v for v in [1, 2, 4] if v in assoc_values]
    
    if not target_bsize:
        target_bsize = bsize_values[:4]
    if not target_assoc:
        target_assoc = assoc_values[:3]
    if not repl_values:
        repl_values = ['l']
    
    print(f"   Block sizes: {target_bsize}")
    print(f"   Associativity: {target_assoc}")
    print(f"   Replacement: {repl_values}")
    
    # 4x3 히트맵 생성 (Application × Replacement Policy)
    best_configs = create_full_heatmap(
        cache_df, target_bsize, target_assoc, repl_values,
        'Graph_5_1_Heatmap.png'
    )
    
    # 요약 테이블 및 막대그래프 생성
    if best_configs:
        create_summary_table(best_configs, 'Graph_5_2_Summary_Table.png')
        create_summary_bar_chart(best_configs, 'Graph_5_3_Summary_Bar.png')


def run_section_6(df):
    """
    ⑥ 최적 System Configuration 탐색 (가산점)
    """
    print("\n📊 Section 6: System-wide Optimization (Bonus)")
    
    if df.empty:
        print("⚠️ No data available for system optimization")
        return
    
    create_comparison_chart(
        df, None,
        title='6. System-wide Optimization: Baseline vs Best Configuration',
        filename='Graph_6_SystemOpt.png'
    )
    
    # 상세 최적화 결과 테이블 출력
    print("\n" + "=" * 80)
    print("📋 Optimal Configuration Summary")
    print("=" * 80)
    
    for bench in BENCHMARKS:
        bench_df = df[df['benchmark'] == bench]
        if bench_df.empty:
            continue
        
        best_row = bench_df.loc[bench_df['sim_IPC'].idxmax()]
        worst_row = bench_df.loc[bench_df['sim_IPC'].idxmin()]
        
        improvement = (best_row['sim_IPC'] - worst_row['sim_IPC']) / worst_row['sim_IPC'] * 100
        
        print(f"\n🔹 {BENCHMARK_DISPLAY[bench]}:")
        print(f"   Baseline IPC: {worst_row['sim_IPC']:.4f}")
        print(f"   Optimized IPC: {best_row['sim_IPC']:.4f}")
        print(f"   Improvement: +{improvement:.2f}%")
        print(f"   Best Config: IALU={best_row.get('res_ialu', '-')}, "
              f"Width={best_row.get('issue_width', '-')}, "
              f"Bpred={best_row.get('bpred', '-')}, "
              f"DL1={best_row.get('dl1_bsize', '-')}B/"
              f"{best_row.get('dl1_assoc', '-')}-way/"
              f"{best_row.get('dl1_repl', '-')}")


def generate_summary_report(df):
    """
    전체 요약 보고서 생성
    """
    print("\n" + "=" * 80)
    print("📝 SIMULATION ANALYSIS SUMMARY REPORT")
    print("=" * 80)
    
    if df.empty:
        print("No data to summarize.")
        return
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total simulations: {len(df)}")
    print(f"   Benchmarks: {', '.join(df['benchmark'].unique())}")
    
    print(f"\n📈 Performance Summary (IPC):")
    for bench in BENCHMARKS:
        bench_df = df[df['benchmark'] == bench]
        if not bench_df.empty:
            print(f"   {BENCHMARK_DISPLAY[bench]:6s}: min={bench_df['sim_IPC'].min():.4f}, "
                  f"max={bench_df['sim_IPC'].max():.4f}, "
                  f"mean={bench_df['sim_IPC'].mean():.4f}")
    
    print(f"\n🔧 Configuration Variations Found:")
    
    if 'res_ialu' in df.columns:
        print(f"   IALU units: {sorted(df['res_ialu'].dropna().unique())}")
    if 'issue_width' in df.columns:
        print(f"   Issue widths: {sorted(df['issue_width'].dropna().unique())}")
    if 'bpred' in df.columns:
        print(f"   Branch predictors: {df['bpred'].dropna().unique().tolist()}")
    if 'dl1_bsize' in df.columns:
        print(f"   DL1 block sizes: {sorted(df['dl1_bsize'].dropna().unique())}")
    if 'dl1_assoc' in df.columns:
        print(f"   DL1 associativity: {sorted(df['dl1_assoc'].dropna().unique())}")
    if 'dl1_repl' in df.columns:
        print(f"   DL1 replacement: {df['dl1_repl'].dropna().unique().tolist()}")


# ==========================================
# 5. 메인 실행
# ==========================================

def main():
    """
    메인 실행 함수
    """
    print("\n" + "🚀" * 40)
    print("\n   SimpleScalar Simulation Analyzer")
    print("   Computer Architecture Project Tool")
    print("\n" + "🚀" * 40 + "\n")
    
    # 1. 시뮬레이션 결과 파일 파싱
    df = parse_all_simulation_files('.')
    
    if df.empty:
        print("\n❌ No valid simulation data found.")
        print("   Please ensure result files (*.txt) are in the current directory.")
        print("   Files should contain SimpleScalar simulation output.")
        return
    
    # 2. 요약 보고서 생성
    generate_summary_report(df)
    
    # 3. 그래프 생성
    print("\n" + "=" * 80)
    print("📊 GENERATING GRAPHS")
    print("=" * 80)
    
    # Section 3: Processor Analysis
    run_section_3_1(df)  # ALU Units
    run_section_3_2(df)  # Issue Width
    run_section_3_3(df)  # Branch Predictor
    
    # Section 4: Cache Analysis
    run_section_4_1(df)  # Block Size
    run_section_4_2(df)  # Associativity
    run_section_4_3(df)  # Replacement Policy
    
    # Section 5: Cache Optimization
    run_section_5(df)
    
    # Section 6: System Optimization
    run_section_6(df)
    
    print("\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\n📁 Generated files:")
    for f in sorted(glob.glob('Graph_*.png')):
        print(f"   ✅ {f}")
    
    print("\n💡 Tips:")
    print("   - Include these graphs in your project report")
    print("   - Remember to analyze WHY the results show these patterns")
    print("   - Consider trade-offs (performance vs hardware cost)")
    print("\n")


if __name__ == "__main__":
    main()
