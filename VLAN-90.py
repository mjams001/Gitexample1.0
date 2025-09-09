import requests
import json
import getpass
import csv
import re
from get_edges import get_edge_info
from petco_store_ips import StoreIPs


user = input('Enter email address here: \n')
password = getpass.getpass()

enterpriseId = 3

login = requests.post('https://vco56-usvi1.velocloud.net/portal/rest/login/enterpriseLogin', json={'username': user, 'password': password})
# print(login.headers)
auth = login.cookies

count = 0

# Call function to get list of edges
result = get_edge_info(enterpriseId,auth)

for edge in result['branches']:
    if count < 1500:
    #if count < 1:
        if 'PETCO' in edge['edgeName']:
            location = re.findall('[0-9][0-9][0-9][0-9]|[0-9][0-9][0-9]', edge["edgeName"])
        #if 'LAB82' in edge['edgeName']:
        #    location = re.findall('[0-9][0-9][0-9][0-9]|[0-9][0-9][0-9]|[0-9][0-9]', edge["edgeName"])
            store = int(location[0])
            count = count + 1
            store_id = edge['edgeId']
            store_name = edge['edgeName']
            get_ips = StoreIPs(store)
            vlan90ip = get_ips.vlan90_gateway

            response = requests.post('https://vco56-usvi1.velocloud.net/portal/rest/edge/getEdgeConfigurationStack', json={'enterpriseId': 3, 'edgeId': int(store_id)}, cookies=auth)
            config_data = json.loads(response.content)

            try:
                edgeSpecificProfile = config_data[0]
                edgeSpecificProfileDeviceSettings = [m for m in edgeSpecificProfile['modules'] if m['name'] == 'deviceSettings'][0]
                edgeSpecificProfileDeviceSettingsData = edgeSpecificProfileDeviceSettings['data']
                moduleId = edgeSpecificProfileDeviceSettings['id']
            
                for i in edgeSpecificProfileDeviceSettingsData['lan']['networks']:
                    if i['vlanId'] == 90 and i['cidrIp'] != vlan90ip:
                        i['cidrIp'] = vlan90ip
                        
                        response_data = { 'enterpriseId': 3,
                        'id': moduleId,
                        '_update': { 'data':  edgeSpecificProfileDeviceSettingsData }}
                        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
                        update = requests.post('https://vco56-usvi1.velocloud.net/portal/rest/configuration/updateConfigurationModule', data=json.dumps(response_data), cookies=auth, headers=headers)
                        if update.status_code == 200:
                            print(f'{store_name} has been updated with correct VLAN 90 configuration')
                        else:
                            print(f'{update.reason,update.status_code},{update.content},{store_name} needs VLAN 90 validation!')                            

            except:
                pass






