import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# 페이지 설정
st.set_page_config(
    page_title="LLM 활용 위험성평가 자동 생성 및 사고 예측",
    page_icon="🛠️",
    layout="wide"
)

# 스타일 적용
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 5px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# 헤더 표시
st.markdown('<div class="main-header">LLM 활용 위험성평가 자동 생성 및 사고 예측</div>', unsafe_allow_html=True)

# 탭 설정
tabs = st.tabs(["시스템 개요", "성능 평가", "사례 분석", "데이터 탐색"])

# 샘플 데이터 (CSV에서 가져온 데이터 하드코딩)
evaluation_results = {
    "index": list(range(1, 27)),
    "작업활동 및 내용": ["Shoring Installation", "In and Out of materials", "Transport / Delivery of scaffold materials", 
                   "Survey and Inspection", "Scaffold Construction", "Plastering work", "Sheet Piling Works",
                   "Mixing of cement plaster", "Excavation and backfilling", "Common", "Plastering work", 
                   "Shoring Installation", "Shoring Installation", "Shoring Installation", "Concrete pouring",
                   "Survey and Inspection", "In and Out of materials", "Excavation and backfilling",
                   "Lifting of scaffold materials", "Survey and Inspection", "Sheet Piling Works",
                   "Excavation and backfilling", "Scaffold Construction", "Concrete pouring", "Common",
                   "Setting up of pumpcrete"],
    "유해위험요인 및 환경측면 영향": ["Fall and collision due to unstable ground", "Overturning of transport vehicle", 
                          "Collision between transport vehicle with others", "Personnel fall while inspecting",
                          "Scaffold collapse due to damage", "Personnel fall from moving along elevated",
                          "Physical injury due to rupture of underground utilities", "Eye splash of chemical during mixing",
                          "Collapse of excavation wall", "Overturning of equipment due to misuse",
                          "Material fall while moving", "Overturning due to carelessness when lifting",
                          "Electric shock", "Collapse of excavation wall and shoring",
                          "Using uninspected power tools", "Excavation collapse", "Material/Equipment fall",
                          "Collapse of excavation wall due to change", "Fall of load due to unsafe slinging",
                          "Personnel falling due to absence of safety", "Collapse of excavation",
                          "Overturning caused by carelessness", "Scaffold collapse due to poor ground",
                          "Personnel fall due to absence of proper working", "Fall caused by inadequate checking",
                          "Collision between pumpcrete/transit mixer"],
    "개선 전 빈도": [3, 3, 3, 2, 2, 3, 1, 1, 3, 3, 3, 2, 2, 2, 3, 2, 2, 2, 3, 3, 3, 2, 3, 3, 2, 3],
    "개선 전 강도": [2, 3, 5, 3, 4, 5, 3, 4, 4, 4, 5, 3, 4, 3, 5, 4, 5, 3, 5, 5, 4, 5, 5, 4, 3, 5],
    "개선 전 T": [6, 9, 15, 6, 8, 15, 3, 4, 12, 12, 15, 6, 8, 6, 15, 8, 10, 6, 15, 15, 12, 10, 15, 12, 6, 15],
    "개선 후 빈도(정답)": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "개선 후 강도(정답)": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "개선 후 T(정답)": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "개선 후 빈도(GPT)": [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "개선 후 강도(GPT)": [2, 2, 3, 2, 2, 2, 3, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "개선 후 T(GPT)": [4, 4, 6, 4, 4, 4, 6, 1, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    "RRR": [33.33, 55.56, 60.00, 33.33, 50.00, 73.33, -100.00, 75.00, 66.67, 66.67, 73.33, 33.33, 25.00, 33.33, 73.33, 50.00, 60.00, 33.33, 73.33, 73.33, 66.67, 60.00, 73.33, 66.67, 33.33, 73.33],
    "Similarity": [0.769, 0.682, 1.000, 0.822, 0.807, 0.883, 0.660, 1.000, 0.858, 0.928, 0.375, 0.634, 0.747, 0.571, 0.690, 0.998, 0.670, 0.975, 0.846, 0.801, 0.987, 0.664, 0.965, 0.621, 0.786, 0.665],
    "improvement_direction_from_original": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(evaluation_results)

# 시스템 개요 탭
with tabs[0]:
    st.markdown('<div class="sub-header">LLM 기반 위험성평가 시스템</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="info-text">
        LLM(Large Language Model)을 활용한 위험성평가 자동화 시스템은 건설 현장의 안전 관리를 혁신적으로 개선합니다:
        
        1. <span class="highlight">작업 내용 입력 시 생성형 AI를 통한 '유해위험요인' 및 '위험 등급' 자동 생성</span>
        2. AI는 건설현장의 기존 위험성평가를 공정별로 구분하고, 해당 유해위험요인을 학습
        3. 학습 후 각 공정별 예상 가능한 유해위험요인을 AI가 스스로 산출하도록 모델 구축 (1월 중 시연)
        4. 자동 생성 기술 개발 완료 후 위험도 기반 사고위험성과 이미지 분석을 통한 사고예측 (5월 중 착수)
        
        이 시스템은 PIMS 및 안전지킴이 등 EHS 플랫폼에 AI 기술 탑재를 통해 통합 사고 예측 프로그램으로 발전 예정입니다.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # AI 위험성평가 프로세스 다이어그램
        st.markdown('<div style="text-align: center; margin-bottom: 10px;"><b>AI 위험성평가 프로세스</b></div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # 프로세스 단계들을 노드로 표시
        steps = ["작업내용 입력", "AI 위험분석", "유해요인 식별", "위험등급 산정", "개선대책 자동생성", "안전조치 적용"]
        
        for i, step in enumerate(steps):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode="markers+text",
                marker=dict(size=30, color="#1E88E5"),
                text=step,
                textposition="bottom center",
                hoverinfo="text",
                name=step
            ))
        
        # 노드 연결선
        for i in range(len(steps)-1):
            fig.add_trace(go.Scatter(
                x=[i, i+1],
                y=[0, 0],
                mode="lines",
                line=dict(width=2, color="#1E88E5"),
                hoverinfo="none",
                showlegend=False
            ))
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            height=300,
            margin=dict(l=20, r=20, t=20, b=100),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 시스템 특징
    st.markdown('<div class="sub-header">시스템 특징 및 구성요소</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### AI 위험성평가 자동생성 시스템 DEMO 개발 (현재)
        - 사우디 3개 및 이집트 PJT 위험성평가 자료 샘플링
        - 위험성평가 학습 과정 포함
        - 대규모 언어 모델(LLM) 기반 위험도 측정 정교화
        - 생성형 AI 기반 위험성 평가 시스템
        - 위험성평가 위험도 자동 생성 플랫폼 (1월)
        """)
    
    with col2:
        st.markdown("""
        #### 사고 예측 및 개선 대책 생성 (계획)
        - 위험도 자동생성 추출 결과 데이터 및 두산 건설EHS / 외부 정보 통합
        - AI의 이미지 분석을 통한 건설현장 CCTV 및 점검 사진 분석
        - 두산 건설현장 통합 AI 사고 예측 프로그램 개발 착수 (5월)
        """)

# 성능 평가 탭
with tabs[1]:
    st.markdown('<div class="sub-header">모델 성능 평가</div>', unsafe_allow_html=True)
    
    # 주요 지표 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="평균 위험 감소율 (RRR)",
            value=f"{df['RRR'].mean():.2f}%",
            delta="목표 대비 +0.60%"
        )
    
    with col2:
        st.metric(
            label="평균 개선방안 유사도",
            value=f"{df['Similarity'].mean():.4f}",
            delta="+0.08"
        )
    
    with col3:
        improvement_ratio = df['improvement_direction_from_original'].mean() * 100
        st.metric(
            label="개선 성공률",
            value=f"{improvement_ratio:.1f}%",
            delta="+1.5%"
        )
    
    with col4:
        st.metric(
            label="정확한 T 값 예측률",
            value="3.85%",
            delta="-1.15%",
            delta_color="inverse"
        )
    
    # 위험 감소율 분포
    st.markdown('<div class="sub-header">위험 감소율(RRR) 분포</div>', unsafe_allow_html=True)
    
    fig = px.histogram(
        df, 
        x="RRR",
        nbins=10,
        title="위험 감소율(RRR) 분포",
        color_discrete_sequence=["#1E88E5"]
    )
    
    fig.update_layout(
        xaxis_title="위험 감소율(%)",
        yaxis_title="빈도",
        bargap=0.1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 여러 지표 비교
    st.markdown('<div class="sub-header">작업 유형별 위험 감소율 및 유사도</div>', unsafe_allow_html=True)
    
    # 작업 유형별 평균 계산
    work_type_stats = df.groupby('작업활동 및 내용').agg({
        'RRR': 'mean',
        'Similarity': 'mean'
    }).reset_index()
    
    # 상위 8개 작업만 선택
    top_works = work_type_stats.sort_values('RRR', ascending=False).head(8)
    
    fig = px.bar(
        top_works,
        x='작업활동 및 내용',
        y=['RRR', 'Similarity'],
        barmode='group',
        title="작업 유형별 주요 성능 지표",
        color_discrete_sequence=["#1E88E5", "#4CAF50"]
    )
    
    fig.update_layout(
        xaxis_title="작업 유형",
        yaxis_title="지표 값",
        legend_title="지표"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 사례 분석 탭
with tabs[2]:
    st.markdown('<div class="sub-header">개선 사례 분석</div>', unsafe_allow_html=True)
    
    # 예시 사례 선택 도구
    example_indices = list(range(len(df)))
    selected_example = st.selectbox(
        "분석할 사례를 선택하세요:",
        example_indices,
        format_func=lambda x: f"{df['작업활동 및 내용'][x]} - {df['유해위험요인 및 환경측면 영향'][x][:30]}..."
    )
    
    # 선택된 사례 표시
    selected_case = df.iloc[selected_example]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### 작업 및 위험 정보
        - **작업활동:** {selected_case['작업활동 및 내용']}
        - **유해위험요인:** {selected_case['유해위험요인 및 환경측면 영향']}
        
        ### 위험도 평가
        - **개선 전 빈도:** {selected_case['개선 전 빈도']}
        - **개선 전 강도:** {selected_case['개선 전 강도']}
        - **개선 전 T값:** {selected_case['개선 전 T']}
        """)
        
        # 위험도 변화 시각화
        risk_data = {
            '단계': ['개선 전', '개선 후 (정답)', '개선 후 (GPT 예측)'],
            'T값': [selected_case['개선 전 T'], selected_case['개선 후 T(정답)'], selected_case['개선 후 T(GPT)']]
        }
        
        risk_df = pd.DataFrame(risk_data)
        
        fig = px.bar(
            risk_df,
            x='단계',
            y='T값',
            color='단계',
            title="위험도(T값) 변화",
            color_discrete_sequence=["#F44336", "#4CAF50", "#2196F3"]
        )
        
        fig.update_layout(
            showlegend=False,
            yaxis_title="위험도 (T값)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        ### GPT 모델 성능
        - **위험 감소율(RRR):** {selected_case['RRR']:.2f}%
        - **개선방안 유사도:** {selected_case['Similarity']:.4f}
        - **위험도 개선 여부:** {'개선됨' if selected_case['improvement_direction_from_original'] == 1 else '개선되지 않음'}
        
        ### 예측 결과
        - **GPT 예측 빈도:** {selected_case['개선 후 빈도(GPT)']}
        - **GPT 예측 강도:** {selected_case['개선 후 강도(GPT)']}
        - **GPT 예측 T값:** {selected_case['개선 후 T(GPT)']}
        """)
        
        # 빈도와 강도의 변화를 히트맵으로 표시
        heat_data = {
            '구분': ['개선 전', '개선 후 (정답)', '개선 후 (GPT)'],
            '빈도': [selected_case['개선 전 빈도'], selected_case['개선 후 빈도(정답)'], selected_case['개선 후 빈도(GPT)']],
            '강도': [selected_case['개선 전 강도'], selected_case['개선 후 강도(정답)'], selected_case['개선 후 강도(GPT)']]
        }
        
        heat_df = pd.DataFrame(heat_data)
        
        fig = go.Figure()
        
        # 개선 전
        fig.add_trace(go.Scatter(
            x=[heat_df['빈도'][0]],
            y=[heat_df['강도'][0]],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='circle',
                line=dict(width=2, color='darkred')
            ),
            name='개선 전'
        ))
        
        # 개선 후 (정답)
        fig.add_trace(go.Scatter(
            x=[heat_df['빈도'][1]],
            y=[heat_df['강도'][1]],
            mode='markers',
            marker=dict(
                size=20,
                color='green',
                symbol='circle',
                line=dict(width=2, color='darkgreen')
            ),
            name='개선 후 (정답)'
        ))
        
        # 개선 후 (GPT)
        fig.add_trace(go.Scatter(
            x=[heat_df['빈도'][2]],
            y=[heat_df['강도'][2]],
            mode='markers',
            marker=dict(
                size=20,
                color='blue',
                symbol='circle',
                line=dict(width=2, color='darkblue')
            ),
            name='개선 후 (GPT)'
        ))
        
        # 배경에 위험도 매트릭스(5x5) 표시
        for i in range(1, 6):
            for j in range(1, 6):
                risk = i * j
                color = 'rgba(255,255,255,0.1)'
                if risk >= 15:
                    color = 'rgba(255,0,0,0.2)'
                elif risk >= 8:
                    color = 'rgba(255,165,0,0.2)'
                elif risk >= 4:
                    color = 'rgba(255,255,0,0.2)'
                else:
                    color = 'rgba(0,128,0,0.2)'
                
                fig.add_shape(
                    type="rect",
                    x0=i-0.5, y0=j-0.5,
                    x1=i+0.5, y1=j+0.5,
                    fillcolor=color,
                    line=dict(width=0),
                    layer='below'
                )
        
        fig.update_layout(
            title="빈도-강도 매트릭스 상 위치 변화",
            xaxis=dict(
                title="빈도",
                range=[0.5, 5.5],
                dtick=1
            ),
            yaxis=dict(
                title="강도",
                range=[0.5, 5.5],
                dtick=1
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 샘플 개선대책 비교 (가상의 예시)
    st.markdown("### 개선대책 비교")
    
    # 샘플 개선대책 하드코딩 (실제 데이터가 없으므로 예시)
    improvement_examples = {
        0: {
            "정답": "1) Check safe lifting requirement for materials 2) Establish proper ground support 3) Apply safety protocols for unstable conditions",
            "GPT": "1) Conduct a thorough assessment of ground conditions before starting work 2) Implement proper stabilization measures 3) Train workers on safety procedures for unstable ground"
        },
        1: {
            "정답": "1) Flat and smooth ground 2) Potholes, uneven surfaces must be repaired 3) Use signalman during movement",
            "GPT": "1) Ensure ground is flat and smooth. 2) Regularly inspect and repair potholes and uneven surfaces 3) Implement mandatory use of signalman for all transport movements"
        },
        7: {
            "정답": "1) Follow instructions based on MSDS 2) Wear safety goggles as per PPE requirements 3) Ensure proper mixing techniques",
            "GPT": "1) Follow instructions based on MSDS 2) Wear safety goggles as per PPE requirements 3) Ensure proper mixing techniques"
        }
    }
    
    example_idx = selected_example
    if example_idx not in improvement_examples:
        example_idx = 0  # 기본값
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        #### 표준 개선대책 (정답)
        ```
        {improvement_examples[example_idx]["정답"]}
        ```
        """)
    
    with col2:
        st.markdown(f"""
        #### AI 생성 개선대책 (GPT)
        ```
        {improvement_examples[example_idx]["GPT"]}
        ```
        """)
    
    # 유사도 시각화
    similarity = selected_case['Similarity']
    st.markdown(f"### 개선대책 유사도: {similarity:.4f}")
    
    progress_color = "#f63366" if similarity < 0.7 else "#02b875" if similarity > 0.9 else "#ffaa00"
    st.markdown(
        f"""
        <div style="border-radius:10px; padding:10px; background-color:{progress_color}20; margin-bottom:10px">
            <div style="background-color:{progress_color}; width:{similarity*100}%; height:30px; border-radius:5px; text-align:center; color:white; line-height:30px; font-weight:bold">
                {similarity*100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# 데이터 탐색 탭
with tabs[3]:
    st.markdown('<div class="sub-header">데이터셋 탐색</div>', unsafe_allow_html=True)
    
    # 전체 데이터 표시
    st.dataframe(df)
    
    # 기본 통계 계산
    st.markdown('<div class="sub-header">기본 통계</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 위험 감소율(RRR) 통계")
        st.write(df['RRR'].describe())
    
    with col2:
        st.markdown("#### 유사도 통계")
        st.write(df['Similarity'].describe())
    
    # 상관관계 시각화
    st.markdown('<div class="sub-header">변수 간 상관관계</div>', unsafe_allow_html=True)
    
    # 상관 관계를 분석할 수치형 변수들
    numeric_cols = ['개선 전 빈도', '개선 전 강도', '개선 전 T', 
                   '개선 후 빈도(GPT)', '개선 후 강도(GPT)', '개선 후 T(GPT)',
                   'RRR', 'Similarity']
    
    correlation = df[numeric_cols].corr()
    
    fig = px.imshow(
        correlation,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="변수 간 상관관계 히트맵"
    )
    
    fig.update_layout(
        height=600,
        width=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 작업 유형 분포
    st.markdown('<div class="sub-header">작업 유형 분포</div>', unsafe_allow_html=True)
    
    work_counts = df['작업활동 및 내용'].value_counts()
    
    fig = px.pie(
        names=work_counts.index,
        values=work_counts.values,
        title="작업 유형별 사례 수",
        hole=0.4
    )
    
    fig.update_layout(
        legend_title="작업 유형"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 푸터
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f0f2f6; border-radius: 10px;">
    <p>© 2023-2025 두산건설 LLM 활용 위험성평가 자동 생성 시스템</p>
    <p style="font-size: 0.8rem;">최신 업데이트: 2025년 3월 23일</p>
</div>
""", unsafe_allow_html=True)