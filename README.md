# SimpleScalar Simulation Analyzer
## 컴퓨터 구조 프로젝트 - 자동 분석 도구

### 📋 개요

이 도구는 SimpleScalar 시뮬레이션 결과를 자동으로 분석하여 프로젝트 보고서에 필요한 그래프를 생성합니다.

### 📁 파일 구성

- `get_ipc_enhanced.py`: 메인 분석 스크립트
- `generate_sample_data.py`: 테스트용 샘플 데이터 생성기
- `Graph_*.png`: 생성된 샘플 그래프들

### 🚀 사용 방법

#### 1. 기본 사용법

```bash
# 시뮬레이션 결과 파일들이 있는 디렉토리로 이동
cd /path/to/your/results

# 스크립트 실행
python3 get_ipc_enhanced.py
```

#### 2. 파일명 규칙 (권장)

스크립트가 자동으로 파일 내용에서 configuration을 파싱하지만, 
파일명을 다음 규칙으로 지정하면 더 정확하게 분류됩니다:

```
result{섹션}_{설정}_{벤치마크}.txt

예시:
- result3_1_1gcc.txt     # Section 3-1, IALU=1, GCC 벤치마크
- result3_2_width2_go.txt   # Section 3-2, Width=2, GO 벤치마크
- result3_3_nottaken_mcf.txt # Section 3-3, Not-Taken predictor, MCF 벤치마크
- result4_1_bsize64_gzip.txt # Section 4-1, Block Size=64, GZIP 벤치마크
- result4_2_assoc4_gcc.txt  # Section 4-2, 4-way associativity, GCC 벤치마크
- result4_3_lru_go.txt    # Section 4-3, LRU replacement, GO 벤치마크
```

### 📊 생성되는 그래프

| 파일명 | 설명 | 프로젝트 섹션 |
|--------|------|--------------|
| `Graph_3_1_ALU.png` | 연산 장치 수에 따른 IPC | ③-1 |
| `Graph_3_2_Width.png` | Super-scalar width에 따른 IPC | ③-2 |
| `Graph_3_3_Bpred.png` | Branch predictor에 따른 IPC | ③-3 |
| `Graph_4_1_BlockSize.png` | Cache block size에 따른 IPC | ④-1 |
| `Graph_4_2_Assoc.png` | Cache associativity에 따른 IPC | ④-2 |
| `Graph_4_3_Replacement.png` | Replacement policy에 따른 IPC | ④-3 |
| `Graph_5_CacheOpt.png` | Cache 최적화 비교 | ⑤ |
| `Graph_5_1_Heatmap.png` | Block Size × Associativity 히트맵 | ⑤ |
| `Graph_6_SystemOpt.png` | 전체 시스템 최적화 비교 | ⑥ |

### 🔧 필요한 시뮬레이션 실행 명령어

프로젝트 PDF에서 요구하는 시뮬레이션들을 실행하세요:

#### Section 3-1: 연산 장치 수 (1, 2, 4)
```bash
# IALU = 1
./sim-outorder -config baseline.cfg -res:ialu 1 -res:imult 1 -res:fpalu 1 -res:fpmult 1 benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result3_1_1gcc.txt

# IALU = 2
./sim-outorder -config baseline.cfg -res:ialu 2 -res:imult 2 -res:fpalu 2 -res:fpmult 2 benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result3_1_2gcc.txt

# IALU = 4
./sim-outorder -config baseline.cfg -res:ialu 4 -res:imult 4 -res:fpalu 4 -res:fpmult 4 benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result3_1_4gcc.txt
```

#### Section 3-2: Issue Width (1, 2, 4)
```bash
./sim-outorder -config baseline.cfg -fetch:ifqsize 1 -decode:width 1 -issue:width 1 -commit:width 1 benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result3_2_width1_gcc.txt
./sim-outorder -config baseline.cfg -fetch:ifqsize 2 -decode:width 2 -issue:width 2 -commit:width 2 benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result3_2_width2_gcc.txt
./sim-outorder -config baseline.cfg -fetch:ifqsize 4 -decode:width 4 -issue:width 4 -commit:width 4 benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result3_2_width4_gcc.txt
```

#### Section 3-3: Branch Predictor
```bash
./sim-outorder -config baseline.cfg -bpred nottaken benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result3_3_nottaken_gcc.txt
./sim-outorder -config baseline.cfg -bpred taken benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result3_3_taken_gcc.txt
./sim-outorder -config baseline.cfg -bpred 1bit benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result3_3_1bit_gcc.txt
./sim-outorder -config baseline.cfg -bpred 3bit benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result3_3_3bit_gcc.txt
```

#### Section 4-1: Block Size (16, 32, 64, 128)
```bash
./sim-outorder -config baseline.cfg -cache:dl1 dl1:128:16:4:l -cache:dl2 ul2:1024:16:4:l -cache:il1 il1:512:16:1:l benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result4_1_bsize16_gcc.txt
./sim-outorder -config baseline.cfg -cache:dl1 dl1:128:32:4:l -cache:dl2 ul2:1024:32:4:l benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result4_1_bsize32_gcc.txt
./sim-outorder -config baseline.cfg -cache:dl1 dl1:128:64:4:l -cache:dl2 ul2:1024:64:4:l benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result4_1_bsize64_gcc.txt
./sim-outorder -config baseline.cfg -cache:dl1 dl1:128:128:4:l -cache:dl2 ul2:1024:128:4:l benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result4_1_bsize128_gcc.txt
```

#### Section 4-2: Associativity (1, 2, 4)
```bash
./sim-outorder -config baseline.cfg -cache:dl1 dl1:128:32:1:l benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result4_2_assoc1_gcc.txt
./sim-outorder -config baseline.cfg -cache:dl1 dl1:128:32:2:l benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result4_2_assoc2_gcc.txt
./sim-outorder -config baseline.cfg -cache:dl1 dl1:128:32:4:l benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result4_2_assoc4_gcc.txt
```

#### Section 4-3: Replacement Policy (LRU, FIFO, Random)
```bash
./sim-outorder -config baseline.cfg -cache:dl1 dl1:128:64:4:l -cache:dl2 ul2:1024:64:4:l benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result4_3_lru_gcc.txt
./sim-outorder -config baseline.cfg -cache:dl1 dl1:128:64:4:f -cache:dl2 ul2:1024:64:4:f benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result4_3_fifo_gcc.txt
./sim-outorder -config baseline.cfg -cache:dl1 dl1:128:64:4:r -cache:dl2 ul2:1024:64:4:r benchmarks/cc1.alpha -O benchmarks/1stmt.i 2> result4_3_random_gcc.txt
```

### 🛠️ 요구사항

```bash
pip install matplotlib numpy pandas
```

### 📌 주의사항

- 결과 파일은 SimpleScalar의 stderr 출력을 리다이렉션(`2>`)해서 생성해야 합니다
- 모든 벤치마크(gcc, go, mcf, gzip)에 대해 동일한 실험을 수행해야 합니다
- baseline.cfg를 기준으로 한 가지 파라미터만 변경하며 실험해야 합니다

---

**Author**: Seokjun Ryu, Computer Architecture Team Project 3조
**Date**: 2025.12
