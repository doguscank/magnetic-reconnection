import os
import wget

"""
1- Navigate to https://omniweb.gsfc.nasa.gov/form/dx1.html
2- Select 'Create File'
3- Select all data checkboxes
4- Get URLs of both data and data description
5- Paste URL below
"""

data_url = "DATA_URL"
data_desc_url = "DATA_DESC_URL"

if not os.path.exists("data"):
    os.mkdir("data")

wget.download(data_desc_url, "data/description.txt")
wget.download(data_url, "data/raw_data.lst")