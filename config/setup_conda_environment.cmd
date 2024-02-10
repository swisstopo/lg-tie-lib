@echo off
set dest_env="D:\envs\conda-py3-tie"
set conda_exe="%ProgramFiles%\ArcGIS\Pro\bin\Python\Scripts\conda.exe"
call %conda_exe% create -p %dest_env% --override-channels -c conda-forge -y --show-channel-urls geopandas scikit-image scipy rasterio shapely mayavi geocube