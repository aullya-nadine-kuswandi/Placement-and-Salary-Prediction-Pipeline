import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

classifier = joblib.load('artifacts/placement_classifier.pkl')
regressor = joblib.load('artifacts/salary_regressor.pkl')


def main():
    st.title("Placement & Salary Prediction")
    
    with st.sidebar:
        st.title("About This Website")
        st.markdown("""
        This website allows you to predict 
        your chances of job placement and 
        estimate your expected salary.""")
        st.markdown("---")

        st.sidebar.info("Whatever the result, keep your spirit up!!!")

    
    tab1, tab2 = st.tabs(["Profile & Academic", "Activity & Lifestyle"])
    
    with tab1:
        st.subheader("Self Profile")
        c1, c2 = st.columns(2)
        with c1:
            gender = st.radio("Gender", ['Male', 'Female'], horizontal=True)
            branch = st.selectbox("Branch", ['CSE', 'ECE', 'IT', 'ME', 'CE'])
        with c2:
            city_tier = st.selectbox("City Tier", ['Tier 1', 'Tier 2', 'Tier 3'])
            family_income_level = st.selectbox("Family Income", ['Low', 'Medium', 'High'])
        
        st.subheader("Academic")
        c1, c2 = st.columns(2)
        with c1:
            cgpa = st.number_input("Current CGPA (0-10)", 0.0, 10.0, 8.0, 0.1)
            tenth_percentage = st.number_input("SSC % (10th)", 0.0, 100.0, 75.0, 0.1)
            twelfth_percentage = st.number_input("HSC % (12th)", 0.0, 100.0, 75.0, 0.1)            
        with c2:
            attendance_percentage = st.number_input("Attendance %", 0.0, 100.0, 80.0, 0.1)
            backlogs = st.number_input("Backlogs", 0, 20, 0)
        
        st.subheader("Skills (1-5)")
        c1, c2 = st.columns(2)
        with c1:
            coding_skill_rating = st.slider("Coding Skill", 1, 5, 3)
            aptitude_skill_rating = st.slider("Aptitude Skill", 1, 5, 3)
        with c2:
            communication_skill_rating = st.slider("Communication Skill", 1, 5, 3)
    
    with tab2:
        st.subheader("Activity")
        c1, c2 = st.columns(2)
        with c1:
            projects_completed = st.number_input("Total Projects", 0, 50, 2)
            hackathons_participated = st.number_input("Total Hackathons Join", 0, 50, 1)
            extracurricular_involvement = st.selectbox("Extracurricular Involvement", ['Low', 'Medium', 'High'])
        with c2:
            internships_completed = st.number_input("Total Internships Completed", 0, 20, 1)
            certifications_count = st.number_input("Certifications", 0, 50, 2)
            part_time_job = st.radio("Have Part Time Job?", ['Yes', 'No'], horizontal=True)
        
        st.subheader("Lifestyle")
        c1, c2 = st.columns(2)
        with c1:
            study_hours_per_day = st.slider("Study Hours/Day", 0.0, 24.0, 3.0, 0.5)
            stress_level = st.slider("Stress Level", 1, 10, 5)
        with c2:
            sleep_hours = st.slider("Sleep Hours", 0.0, 24.0, 7.0, 0.5)
            internet_access = st.radio("Have Internet Access?", ['Yes', 'No'], horizontal=True)
    
    st.markdown("---")
    
    submitted = st.button("Make Prediction", use_container_width=True, type="primary")
    
    if submitted:
        features = {
            'gender': gender, 'branch': branch, 'cgpa': cgpa,
            'tenth_percentage': tenth_percentage, 'twelfth_percentage': twelfth_percentage,
            'backlogs': backlogs, 'study_hours_per_day': study_hours_per_day,
            'attendance_percentage': attendance_percentage,
            'projects_completed': projects_completed,
            'internships_completed': internships_completed,
            'coding_skill_rating': coding_skill_rating,
            'communication_skill_rating': communication_skill_rating,
            'aptitude_skill_rating': aptitude_skill_rating,
            'hackathons_participated': hackathons_participated,
            'certifications_count': certifications_count,
            'sleep_hours': sleep_hours, 'stress_level': stress_level,
            'part_time_job': part_time_job,
            'family_income_level': family_income_level, 'city_tier': city_tier,
            'internet_access': internet_access,
            'extracurricular_involvement': extracurricular_involvement
        }
        
        df = pd.DataFrame([features])
        
        placement_pred = classifier.predict(df)[0]
        placement_proba = classifier.predict_proba(df)[0]
        proba_placed = float(placement_proba[1])
        
        if placement_pred == 1:
            salary_pred = max(0.0, float(regressor.predict(df)[0]))
        else:
            salary_pred = 0.0
        
        st.session_state['prediction_made'] = True
        st.session_state['features'] = features
        st.session_state['placement_pred'] = int(placement_pred)
        st.session_state['proba_placed'] = proba_placed
        st.session_state['salary_pred'] = salary_pred
    
    if st.session_state.get('prediction_made', False):
        features = st.session_state['features']
        placement_pred = st.session_state['placement_pred']
        proba_placed = st.session_state['proba_placed']
        salary_pred = st.session_state['salary_pred']
        
        st.subheader("Prediction")

        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            if placement_pred == 1:
                st.success("### PLACED")
            else:
                st.error("### NOT PLACED")
            
            st.metric(label="Estimated Salary", value=f"{salary_pred:.2f} LPA")
        
        with col_right:
            col_gauge, col_radar = st.columns(2)
            
            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba_placed * 100,
                    title={'text': "Placement Probability", 'font': {'size': 16}},
                    number={'suffix': "%", 'font': {'size': 32}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green" if proba_placed >= 0.5 else "red"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightpink"},
                            {'range': [40, 60], 'color': "lightyellow"},
                            {'range': [60, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75, 'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_radar:
                radar_data = {
                    'CGPA': (features['cgpa'] / 10) * 100,
                    'Study Hours': min(features['study_hours_per_day'] / 10, 1) * 100,
                    'Coding': (features['coding_skill_rating'] / 5) * 100,
                    'Communication': (features['communication_skill_rating'] / 5) * 100,
                    'Aptitude': (features['aptitude_skill_rating'] / 5) * 100,
                    'Projects': min(features['projects_completed'] / 10, 1) * 100,
                    'Internships': min(features['internships_completed'] / 5, 1) * 100,
                    'Certifications': min(features['certifications_count'] / 10, 1) * 100,
                }
                categories = list(radar_data.keys())
                values = list(radar_data.values())
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='Profil',
                    line=dict(color='#00CC96', width=2),
                    fillcolor='rgba(0, 204, 150, 0.3)'
                ))
                fig_radar.update_layout(
                    title={'text': "Your Profile", 'font': {'size': 16}},
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    height=320,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        
        st.header("What Aspect Really Impact Your Placement?")
        
        top_n = st.slider(
            "Show Top Aspect:",
            min_value=3, max_value=15, value=5, step=1
        )
        
        preprocessor = classifier.named_steps['preprocessing']
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'ohe':
                ohe = transformer.named_steps['cat_onehot_encoder']
                feature_names.extend(ohe.get_feature_names_out(cols))
            elif name == 'ordinal':
                feature_names.extend(cols)
        
        coefs_clf = classifier.named_steps['classifier'].coef_[0]
        imp_clf = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs_clf,
            'Abs_Coefficient': np.abs(coefs_clf)
        }).sort_values('Abs_Coefficient', ascending=False).head(top_n)
        
        fig_clf = px.bar(
            imp_clf.sort_values('Coefficient'),
            x='Coefficient', y='Feature', orientation='h',
            color='Coefficient',
            color_continuous_scale=['#EF553B', '#FFFFFF', '#00CC96'],
            color_continuous_midpoint=0,
            text=imp_clf.sort_values('Coefficient')['Coefficient'].apply(lambda x: f"{x:.3f}")
        )
        fig_clf.update_traces(textposition='outside')
        fig_clf.update_layout(
            height=max(350, top_n * 45), showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_clf, use_container_width=True)

if __name__ == '__main__':
    main()