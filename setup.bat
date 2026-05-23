@echo off
echo Installing dependencies...
pip install streamlit pandas numpy scikit-learn scipy xgboost
echo.
echo Installation complete!
echo.
echo To run the app, execute: streamlit run app.py
pause
