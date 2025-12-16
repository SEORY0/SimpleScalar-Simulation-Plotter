#!/usr/bin/env python3
"""
Section 6: Final System Optimization Comparison Graph
- Section 3, 4, 5 최적 IPC vs Section 6 최적 IPC 비교
"""

import matplotlib.pyplot as plt
import numpy as np

# 벤치마크 설정
BENCHMARKS = ['gcc', 'go', 'mcf', 'gzip']
BENCHMARK_DISPLAY = {'gcc': 'GCC', 'go': 'GO', 'mcf': 'MCF', 'gzip': 'GZIP'}

# 각 Section별 최적 IPC 값 (하드코딩)
IPC_DATA = {
    'section3': {'gcc': 1.235, 'go': 1.333, 'mcf': 0.632, 'gzip': 1.844},
    'section4': {'gcc': 1.2896, 'go': 1.3412, 'mcf': 0.6691, 'gzip': 1.9061},
    'section5': {'gcc': 1.2896, 'go': 1.3412, 'mcf': 0.6691, 'gzip': 1.9061},
    'section6': {'gcc': 1.546, 'go': 1.347, 'mcf': 0.682, 'gzip': 2.943},
}

# 색상 팔레트
COLORS = {
    'section3': '#E07A5F',   # 빨간색 계열
    'section4': '#F2CC8F',   # 노란색 계열
    'section5': '#81B29A',   # 초록색 계열
    'section6': '#3D405B',   # 진한 남색
}


def create_section_comparison_chart(filename='Graph_6_1_Section_Comparison.png'):
    """
    Graph 6-1: Section 3, 4, 5, 6 IPC 비교 그룹 막대그래프
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sections = ['section3', 'section4', 'section5', 'section6']
    section_labels = ['Section 3\n(Processor Opt)', 'Section 4\n(Cache Analysis)', 
                      'Section 5\n(Cache Opt)', 'Section 6\n(Full Opt)']
    
    x = np.arange(len(BENCHMARKS))
    width = 0.2
    
    for i, (section, label) in enumerate(zip(sections, section_labels)):
        ipcs = [IPC_DATA[section][b] for b in BENCHMARKS]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, ipcs, width, label=label, 
                     color=COLORS[section], edgecolor='white', linewidth=1.5)
        
        # 값 표시
        for bar, ipc in zip(bars, ipcs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{ipc:.3f}', ha='center', va='bottom',
                   fontweight='bold', fontsize=8)
    
    ax.set_xlabel('Benchmark', fontweight='bold', fontsize=12)
    ax.set_ylabel('IPC (Instructions Per Cycle)', fontweight='bold', fontsize=12)
    ax.set_title('6-1. Performance Comparison: Section 3 vs 4 vs 5 vs 6',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_DISPLAY[b] for b in BENCHMARKS], fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Y축 범위 설정
    all_ipcs = [IPC_DATA[s][b] for s in sections for b in BENCHMARKS]
    ax.set_ylim(0, max(all_ipcs) * 1.15)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


def create_improvement_vs_section3(filename='Graph_6_2_Improvement_vs_Section3.png'):
    """
    Graph 6-2: Section 3 대비 각 Section 개선율 비교
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sections_to_compare = ['section4', 'section5', 'section6']
    section_labels = ['Section 4\n(Cache Analysis)', 'Section 5\n(Cache Opt)', 'Section 6\n(Full Opt)']
    colors = [COLORS['section4'], COLORS['section5'], COLORS['section6']]
    
    x = np.arange(len(BENCHMARKS))
    width = 0.25
    
    for i, (section, label, color) in enumerate(zip(sections_to_compare, section_labels, colors)):
        improvements = []
        for b in BENCHMARKS:
            baseline = IPC_DATA['section3'][b]
            current = IPC_DATA[section][b]
            imp = ((current - baseline) / baseline) * 100
            improvements.append(imp)
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, improvements, width, label=label, 
                     color=color, edgecolor='white', linewidth=1.5)
        
        # 값 표시
        for bar, imp in zip(bars, improvements):
            va = 'bottom' if imp >= 0 else 'top'
            offset_y = 1 if imp >= 0 else -1
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset_y,
                   f'{imp:+.1f}%', ha='center', va=va,
                   fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Benchmark', fontweight='bold', fontsize=12)
    ax.set_ylabel('Improvement Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title('6-2. Performance Improvement vs Section 3 (Processor Optimization)',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_DISPLAY[b] for b in BENCHMARKS], fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


def create_improvement_vs_section4(filename='Graph_6_3_Improvement_vs_Section4.png'):
    """
    Graph 6-3: Section 4 대비 Section 5, 6 개선율 비교
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sections_to_compare = ['section5', 'section6']
    section_labels = ['Section 5\n(Cache Opt)', 'Section 6\n(Full Opt)']
    colors = [COLORS['section5'], COLORS['section6']]
    
    x = np.arange(len(BENCHMARKS))
    width = 0.3
    
    for i, (section, label, color) in enumerate(zip(sections_to_compare, section_labels, colors)):
        improvements = []
        for b in BENCHMARKS:
            baseline = IPC_DATA['section4'][b]
            current = IPC_DATA[section][b]
            imp = ((current - baseline) / baseline) * 100
            improvements.append(imp)
        
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, improvements, width, label=label, 
                     color=color, edgecolor='white', linewidth=1.5)
        
        # 값 표시
        for bar, imp in zip(bars, improvements):
            va = 'bottom' if imp >= 0 else 'top'
            offset_y = 1 if imp >= 0 else -1
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset_y,
                   f'{imp:+.1f}%', ha='center', va=va,
                   fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Benchmark', fontweight='bold', fontsize=12)
    ax.set_ylabel('Improvement Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title('6-3. Performance Improvement vs Section 4 (Cache Analysis)',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_DISPLAY[b] for b in BENCHMARKS], fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


def create_improvement_vs_section5(filename='Graph_6_4_Improvement_vs_Section5.png'):
    """
    Graph 6-4: Section 5 대비 Section 6 개선율 비교
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    improvements = []
    for b in BENCHMARKS:
        baseline = IPC_DATA['section5'][b]
        current = IPC_DATA['section6'][b]
        imp = ((current - baseline) / baseline) * 100
        improvements.append(imp)
    
    x = np.arange(len(BENCHMARKS))
    colors = ['#E07A5F', '#3D9970', '#F4D35E', '#2E86AB']
    
    bars = ax.bar(x, improvements, color=colors, edgecolor='white', linewidth=2)
    
    # 값 표시
    for bar, imp in zip(bars, improvements):
        va = 'bottom' if imp >= 0 else 'top'
        offset_y = 1 if imp >= 0 else -1
        color = '#2E86AB' if imp >= 0 else '#C73E1D'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset_y,
               f'{imp:+.1f}%', ha='center', va=va,
               fontweight='bold', fontsize=12, color=color)
    
    # 평균 개선율
    avg_imp = np.mean(improvements)
    ax.axhline(y=avg_imp, color='#3D405B', linestyle='--', linewidth=2, label=f'Average: {avg_imp:+.1f}%')
    
    ax.set_xlabel('Benchmark', fontweight='bold', fontsize=12)
    ax.set_ylabel('Improvement Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title('6-4. Section 6 (Full Optimization) vs Section 5 (Cache Optimization)',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_DISPLAY[b] for b in BENCHMARKS], fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


def create_summary_table(filename='Graph_6_5_Summary_Table.png'):
    """
    Graph 6-5: 전체 요약 테이블
    """
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis('off')
    
    # 테이블 데이터 준비
    headers = ['Benchmark', 
               'Section 3\n(Processor)', 
               'Section 4\n(Cache)', 
               'Section 5\n(Cache Opt)', 
               'Section 6\n(Full Opt)',
               'Δ vs S3',
               'Δ vs S4',
               'Δ vs S5']
    
    table_data = []
    for b in BENCHMARKS:
        s3 = IPC_DATA['section3'][b]
        s4 = IPC_DATA['section4'][b]
        s5 = IPC_DATA['section5'][b]
        s6 = IPC_DATA['section6'][b]
        
        imp_vs_s3 = ((s6 - s3) / s3) * 100
        imp_vs_s4 = ((s6 - s4) / s4) * 100
        imp_vs_s5 = ((s6 - s5) / s5) * 100
        
        row = [
            BENCHMARK_DISPLAY[b],
            f'{s3:.4f}',
            f'{s4:.4f}',
            f'{s5:.4f}',
            f'{s6:.4f}',
            f'{imp_vs_s3:+.1f}%',
            f'{imp_vs_s4:+.1f}%',
            f'{imp_vs_s5:+.1f}%'
        ]
        table_data.append(row)
    
    # 평균 행 추가
    avg_s3 = np.mean([IPC_DATA['section3'][b] for b in BENCHMARKS])
    avg_s4 = np.mean([IPC_DATA['section4'][b] for b in BENCHMARKS])
    avg_s5 = np.mean([IPC_DATA['section5'][b] for b in BENCHMARKS])
    avg_s6 = np.mean([IPC_DATA['section6'][b] for b in BENCHMARKS])
    
    avg_imp_s3 = np.mean([((IPC_DATA['section6'][b] - IPC_DATA['section3'][b]) / IPC_DATA['section3'][b]) * 100 for b in BENCHMARKS])
    avg_imp_s4 = np.mean([((IPC_DATA['section6'][b] - IPC_DATA['section4'][b]) / IPC_DATA['section4'][b]) * 100 for b in BENCHMARKS])
    avg_imp_s5 = np.mean([((IPC_DATA['section6'][b] - IPC_DATA['section5'][b]) / IPC_DATA['section5'][b]) * 100 for b in BENCHMARKS])
    
    avg_row = [
        'Average',
        f'{avg_s3:.4f}',
        f'{avg_s4:.4f}',
        f'{avg_s5:.4f}',
        f'{avg_s6:.4f}',
        f'{avg_imp_s3:+.1f}%',
        f'{avg_imp_s4:+.1f}%',
        f'{avg_imp_s5:+.1f}%'
    ]
    table_data.append(avg_row)
    
    # 테이블 생성
    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # 헤더 스타일
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3D405B')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 행 색상
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i == len(table_data):  # 평균 행
                table[(i, j)].set_facecolor('#E8E8E8')
                table[(i, j)].set_text_props(fontweight='bold')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    ax.set_title('6-5. System Optimization Summary: IPC Comparison & Improvement Rates',
                 fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


def create_waterfall_chart(filename='Graph_6_6_Optimization_Waterfall.png'):
    """
    Graph 6-6: Application별 최적화 단계 Waterfall 차트
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#E07A5F', '#F2CC8F', '#81B29A', '#3D405B']
    stage_labels = ['Section 3\n(Processor)', 'Section 4\n(Cache)', 'Section 5\n(Cache Opt)', 'Section 6\n(Full Opt)']
    sections = ['section3', 'section4', 'section5', 'section6']
    
    for idx, bench in enumerate(BENCHMARKS):
        ax = axes[idx]
        
        ipcs = [IPC_DATA[s][bench] for s in sections]
        
        x = np.arange(len(ipcs))
        bars = ax.bar(x, ipcs, color=colors, edgecolor='white', linewidth=2)
        
        # 값 표시
        for bar, ipc in zip(bars, ipcs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{ipc:.4f}', ha='center', va='bottom',
                   fontweight='bold', fontsize=10)
        
        # 전체 개선율 (Section 3 → Section 6)
        total_imp = ((ipcs[3] - ipcs[0]) / ipcs[0]) * 100
        ax.annotate('', xy=(3, ipcs[3]), xytext=(0, ipcs[0]),
                   arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2))
        
        mid_x = 1.5
        mid_y = (ipcs[0] + ipcs[3]) / 2
        color = '#2E86AB' if total_imp >= 0 else '#C73E1D'
        ax.text(mid_x, mid_y + 0.05, f'Total: {total_imp:+.1f}%',
               ha='center', va='bottom', fontweight='bold', fontsize=11,
               color=color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=color, alpha=0.9))
        
        ax.set_title(f'{BENCHMARK_DISPLAY[bench]}', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels, fontsize=8)
        ax.set_ylabel('IPC', fontweight='bold', fontsize=10)
        ax.set_ylim(0, max(ipcs) * 1.25)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle('6-6. Optimization Progress: Section 3 → 4 → 5 → 6 (per Application)',
                 fontweight='bold', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Generated: {filename}")
    plt.close()


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("  Section 6: Final Optimization Comparison Graphs")
    print("=" * 60)
    
    print("\n📊 IPC Data:")
    print("-" * 60)
    print(f"{'Benchmark':<10} {'Section3':>10} {'Section4':>10} {'Section5':>10} {'Section6':>10}")
    print("-" * 60)
    for b in BENCHMARKS:
        print(f"{BENCHMARK_DISPLAY[b]:<10} {IPC_DATA['section3'][b]:>10.4f} {IPC_DATA['section4'][b]:>10.4f} {IPC_DATA['section5'][b]:>10.4f} {IPC_DATA['section6'][b]:>10.4f}")
    print("-" * 60)
    
    print("\nGenerating graphs...")
    print("-" * 60)
    
    # 그래프 생성
    create_section_comparison_chart()
    create_improvement_vs_section3()
    create_improvement_vs_section4()
    create_improvement_vs_section5()
    create_summary_table()
    create_waterfall_chart()
    
    print("\n" + "=" * 60)
    print("🎉 All graphs generated successfully!")
    print("=" * 60)
    print("\n📁 Generated files:")
    print("   ✅ Graph_6_1_Section_Comparison.png")
    print("   ✅ Graph_6_2_Improvement_vs_Section3.png")
    print("   ✅ Graph_6_3_Improvement_vs_Section4.png")
    print("   ✅ Graph_6_4_Improvement_vs_Section5.png")
    print("   ✅ Graph_6_5_Summary_Table.png")
    print("   ✅ Graph_6_6_Optimization_Waterfall.png")


if __name__ == '__main__':
    main()
