This script will be used to do the following: 
    1. Reconfigure VLAN 90 for any branch edge device that is currently misconfigured. This current address will be validated against the correct IP (pulled from the petco_store_ip function) and will be updated for the module and pushed to the edge. 

Imports/Dependencies:
    1. get_edges function, using velocloud script to classify edges into dictionaries for branches and hubs
    2. petco_store_ips function, using python class logic to return ip information for specific store devices

This Change was peer-reviewed by Haley prior to CAB, 10/27. 




