import requests
import json
import re

def get_edge_info(enterpriseId,auth):
    hubs = []
    branches = []

    response = requests.post('https://vco56-usvi1.velocloud.net/portal/rest/enterprise/getEnterpriseEdges', json={"id": 0,"enterpriseId": int(enterpriseId),"with": ["links"]}, cookies=auth)
    # print(response.status_code)
    data = (response.content)
    data_obj = json.loads(data)

    for edge in data_obj:
        enterpriseId = edge['enterpriseId']
        edgeId = edge['id']
        siteId = edge['siteId']
        softwareVersion = edge['softwareVersion']
        serialNumber = edge['serialNumber']
        deviceType = edge['deviceFamily']
        edgeName = edge['name']
        edgeStatus = edge['edgeState']
        lastSeen = edge['lastContact']
        primaryInterface = None
        primaryPublicIp = None
        primaryIsp = None
        primaryLinkState = None
        secondaryInterface = None
        secondaryPublicIp = None
        secondaryIsp = None
        secondaryLinkState = None
        tertiaryInterface = None
        tertiaryPublicIp = None
        tertiaryIsp = None
        tertiaryLinkState = None
        wanInterface = None
        publicIp = None
        isp = None
        linkState = None
        
        edgeName = re.sub(' /.*/', ',', edgeName)
        edgeName = re.sub(',  ', ', ', edgeName)

        if edge['isHub'] == False:
            enterpriseId = edge['enterpriseId']
            for i in edge['links']:
                if i['interface'] == 'GE3' or i['interface'] == 'GE1' and (i['state'] == "STABLE"):
                    primaryInterface = i['interface']
                    primaryPublicIp = i['ipAddress']
                    primaryIsp = i['displayName']
                    primaryLinkState = i['state']
                elif i['interface'] == 'GE4' or i['interface'] == 'GE2' and (i['state'] == "STABLE"):
                    secondaryInterface = i['interface']
                    secondaryPublicIp = i['ipAddress']
                    secondaryIsp = i['displayName']
                    secondaryLinkState = i['state']
                elif i['interface'] == 'GE5' and (i['state'] == "STABLE"):
                    tertiaryInterface = i['interface']
                    tertiaryPublicIp = i['ipAddress']
                    tertiaryIsp = i['displayName']
                    tertiaryLinkState = i['state']
            branches.append({"enterpriseId": enterpriseId, "edgeId": edgeId, "siteId": siteId, "edgeName": edgeName, "softwareVersion": softwareVersion, "serialNumber": serialNumber, "deviceType": deviceType, "edgeStatus": edgeStatus, "lastSeen": lastSeen, "primaryInterface": primaryInterface, "primaryPublicIp": primaryPublicIp, "primaryIsp": primaryIsp, "primaryLinkState": primaryLinkState, "secondaryInterface": secondaryInterface, "secondaryPublicIp": secondaryPublicIp, "secondaryIsp": secondaryIsp, "secondaryLinkState": secondaryLinkState, "tertiaryInterface": tertiaryInterface, "tertiaryPublicIp": tertiaryPublicIp, "tertiaryIsp": tertiaryIsp, "tertiaryLinkState": tertiaryLinkState})

        else:
            enterpriseId = edge['enterpriseId']
            for i in edge['links']:
                if i['state'] == "STABLE":
                    wanInterface = i['interface']
                    publicIp = i['ipAddress']
                    isp = i['displayName']
                    linkState = i['state']
                else:
                    pass
            hubs.append({"enterpriseId": enterpriseId, "edgeId": edgeId, "siteId": siteId, "softwareVersion": softwareVersion, "serialNumber": serialNumber, "deviceType": deviceType, "edgeName": edgeName, "edgeStatus": edgeStatus, "lastSeen": lastSeen, "wanInterface": wanInterface, "publicIp": publicIp, "isp": isp, "linkState": linkState})
    return{"branches":branches, "hubs":hubs}