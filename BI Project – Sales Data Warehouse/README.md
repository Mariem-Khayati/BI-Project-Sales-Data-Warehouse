# SalesDW ETL + Sales Forecasting

## Overview
This project contains:
- An ETL pipeline built with SSIS to populate a Sales Data Warehouse (SalesDW)
- SQL Server (SSMS) scripts to query the Data Warehouse and extract data for analysis
- A Machine Learning module for monthly sales forecasting

## Structure
- Sales_DW_ETL/ : SSIS packages (.dtsx, .dtproj)
- Sales DW (Cube Creation)/ : Data Warehouse and cube scripts
- Data visualization.pbix : Power BI dashboard
- Sales_Forecasting_ML/ : Python-based sales forecasting (ARIMA / SARIMA)

## How to run
1. Open the SSIS solution and configure Connection Managers
2. Execute the SSIS packages to load the DW
3. Run SQL scripts in `sql/`
4. Use `data/sales_final.csv` for forecasting
