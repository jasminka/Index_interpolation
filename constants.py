# Paths
# SATELLITE_IMGS_PATH = r"E:\work\ARRS suša\sentinel_2"
SATELLITE_IMGS_PATH = r"E:\work\ARRS susa\podatki\ARRS_Susa\Sentinel-2_atm_10m_D96_2017"
GERK_SHAPEFILES_PATH = r"E:\work\ARRS susa\podatki\ARRS_Susa\ARSKTRP_ZV-test_section"

KMRS_LIST = {
    "001": "pšenica",
    "007": "tritikala",
    "008": "oves",
    "009": "ječmen",
    "014": "oljna ogrščica",
}

# Number of parallel jobs
N_JOBS = 5

# Postgres connection
CONN_HOST = "127.0.0.1"
CONN_PORT = "5433"
CONN_DATABASE = "drought_interpolation_test"
CONN_USER = "postgres"
CONN_PASSWORD = "postgres"
