import streamlit as st
import pandas as pd
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="LLM 활용 위험성평가 자동 생성 및 사고 예측",
    page_icon="🛠️",
    layout="wide"
)

# 헤더 표시
st.title("LLM 활용 위험성평가 자동 생성 및 사고 예측")

# 탭 설정
tabs = st.tabs(["시스템 개요", "성능 평가", "사례 분석", "데이터 탐색"])

# 시스템 개요 탭
with tabs[0]:
    st.header("LLM 기반 위험성평가 시스템")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.write("""
        LLM(Large Language Model)을 활용한 위험성평가 자동화 시스템은 건설 현장의 안전 관리를 혁신적으로 개선합니다:
        
        1. 작업 내용 입력 시 생성형 AI를 통한 '유해위험요인' 및 '위험 등급' 자동 생성
        2. AI는 건설현장의 기존 위험성평가를 공정별로 구분하고, 해당 유해위험요인을 학습
        3. 학습 후 각 공정별 예상 가능한 유해위험요인을 AI가 스스로 산출하도록 모델 구축 (1월 중 시연)
        4. 자동 생성 기술 개발 완료 후 위험도 기반 사고위험성과 이미지 분석을 통한 사고예측 (5월 중 착수)
        
        이 시스템은 PIMS 및 안전지킴이 등 EHS 플랫폼에 AI 기술 탑재를 통해 통합 사고 예측 프로그램으로 발전 예정입니다.
        """)
    
    with col2:
        st.subheader("AI 위험성평가 프로세스")
        steps = ["작업내용 입력", "AI 위험분석", "유해요인 식별", "위험등급 산정", "개선대책 자동생성", "안전조치 적용"]
        for step in steps:
            st.write(f"- {step}")
    
    # 시스템 특징
    st.header("시스템 특징 및 구성요소")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AI 위험성평가 자동생성 시스템 DEMO 개발 (현재)")
        st.write("""
        - 사우디 3개 및 이집트 PJT 위험성평가 자료 샘플링
        - 위험성평가 학습 과정 포함
        - 대규모 언어 모델(LLM) 기반 위험도 측정 정교화
        - 생성형 AI 기반 위험성 평가 시스템
        - 위험성평가 위험도 자동 생성 플랫폼 (1월)
        """)
    
    with col2:
        st.subheader("사고 예측 및 개선 대책 생성 (계획)")
        st.write("""
        - 위험도 자동생성 추출 결과 데이터 및 두산 건설EHS / 외부 정보 통합
        - AI의 이미지 분석을 통한 건설현장 CCTV 및 점검 사진 분석
        - 두산 건설현장 통합 AI 사고 예측 프로그램 개발 착수 (5월)
        """)

# 푸터
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f0f2f6; border-radius: 10px;">
    <p>© 2023-2025 두산건설 LLM 활용 위험성평가 자동 생성 시스템</p>
    <p style="font-size: 0.8rem;">최신 업데이트: 2025년 3월 23일</p>
</div>
""", unsafe_allow_html=True)
