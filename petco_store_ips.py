INFRA_B_OFFSET = 1
GUEST_B_OFFSET = 101
STORE_B_OFFSET = 201


class StoreIPs:
    """
    VLAN 1      Static VLAN (no DHCP) for: BOC, registers, and switch management
    VLAN 32     Wireless/wired VLAN newly allocated for Vet Hospital devices on PetcoCare SSID.
                  Also currently supports IPET PSK ssid for timeclocks
    VLAN 80     AP Mgmt VLAN for store access points to reach on-prem/cloud controller
    VLAN 90     Velo-Segmented (Guest Segment)  for wireless guest/partner traffic.
    VLAN 93     Velo-Segmented (Guest Segment) for wired vendor traffic
    VLAN 128    Wireless/wired VLAN allocated for store wireless devices, phones, and printers.
                  PetcoStore is the cert-based SSID and SMU9000/Spectra are the legacy PSK SSIDs
    VLAN 248    Static DVR VLAN used for store DVR/Security cameras
    VLAN 250    Static VLAN (no DHCP) for: grooming PC, kiosks, and LMS machines
    """

    def __init__(self, store):
        self.store = int(store)
        self._a = 10
        self._boffset = int(store) // 254
        self._c = int(store) % 254 + 1

    def _make_ip(self, b, d):
        return f"{self._a}.{self._boffset + b}.{self._c}.{d}"

    def _make_ip_range(self, b, drange):
        return [self._make_ip(b, d) for d in drange]

    def _make_cidr(self, b, d, mask):
        return self._make_ip(b, d) + mask

    ######################################################################
    # Infra IPs
    ######################################################################
    vlan1_hosts_list = property(lambda self: self._make_ip_range(INFRA_B_OFFSET, range(1, 253 + 1)))
    vlan1_gateway = property(lambda self: self._make_ip(INFRA_B_OFFSET, 254))
    vlan1_subnet = property(lambda self: self._make_cidr(INFRA_B_OFFSET, 0, "/24"))

    ######################################################################
    # Store IPS
    ######################################################################
    vlan32_hosts = property(lambda self: self._make_ip_range(STORE_B_OFFSET, range(33, 61 + 1)))
    vlan32_gateway = property(lambda self: self._make_ip(STORE_B_OFFSET, 62))
    vlan32_subnet = property(lambda self: self._make_cidr(STORE_B_OFFSET, 32, "/27"))
    vlan32_static_list = property(lambda self: self._make_ip_range(STORE_B_OFFSET, range(57, 61 + 1)))
    vlan32_ssids_list = ["PetcoCare (802.1X)", "IPET (PSK)"]

    vlan80_hosts = property(lambda self: self._make_ip_range(STORE_B_OFFSET, range(82, 94 + 1)))
    vlan80_gateway = property(lambda self: self._make_ip(STORE_B_OFFSET, 1))
    vlan80_subnet = property(lambda self: self._make_cidr(STORE_B_OFFSET, 80, "/28"))

    vlan128_hosts = property(lambda self: self._make_ip_range(STORE_B_OFFSET, range(130, 190 + 1)))
    vlan128_gateway = property(lambda self: self._make_ip(STORE_B_OFFSET, 129))
    vlan128_subnet = property(lambda self: self._make_cidr(STORE_B_OFFSET, 128, "/26"))
    vlan128_ssids_list = ["Spectra/SMU9000 (PSK)", "PetcoStore (802.1X)"]

    vlan248_hosts = property(lambda self: self._make_ip_range(STORE_B_OFFSET, range(249, 249 + 1)))
    vlan248_gateway = property(lambda self: self._make_ip(STORE_B_OFFSET, 250))
    vlan248_subnet = property(lambda self: self._make_cidr(STORE_B_OFFSET, 248, "/30"))

    vlan250_hosts = property(lambda self: self._make_ip_range(STORE_B_OFFSET, range(1, 29 + 1)))
    vlan250_gateway = property(lambda self: self._make_ip(STORE_B_OFFSET, 30))
    vlan250_subnet = property(lambda self: self._make_cidr(STORE_B_OFFSET, 0, "/27"))

    ######################################################################
    # Guest Segment IPs
    ######################################################################
    vlan90_hosts = property(lambda self: self._make_ip_range(GUEST_B_OFFSET, range(2, 62 + 1)))
    vlan90_gateway = property(lambda self: self._make_ip(GUEST_B_OFFSET, 1))
    vlan90_subnet = property(lambda self: self._make_cidr(GUEST_B_OFFSET, 0, "/26"))
    vlan90_ssids_list = ["PetcoGuests (Open)"]

    vlan93_hosts = property(lambda self: self._make_ip_range(GUEST_B_OFFSET, range(130, 134 + 1)))
    vlan93_gateway = property(lambda self: self._make_ip(GUEST_B_OFFSET, 129))
    vlan93_subnet = property(lambda self: self._make_cidr(GUEST_B_OFFSET, 128, "/29"))

    ######################################################################
    # Device IPs
    ######################################################################
    bo_server = property(lambda self: self._make_ip(INFRA_B_OFFSET, 1))
    register_list = property(lambda self: self._make_ip_range(INFRA_B_OFFSET, range(10, 17 + 1)))
    vce_mgmt_ip = property(lambda self: self._make_ip(INFRA_B_OFFSET, 99))
    switch = property(lambda self: self._make_ip(INFRA_B_OFFSET, 253))

    zc = property(lambda self: self._make_ip(STORE_B_OFFSET, 130))
    lms_mac_mini = property(lambda self: self._make_ip(STORE_B_OFFSET, 10))
    grooming_pc = property(lambda self: self._make_ip(STORE_B_OFFSET, 11))
    backoffice_printer = property(lambda self: self._make_ip(STORE_B_OFFSET, 185))
    grooming_printer = property(lambda self: self._make_ip(STORE_B_OFFSET, 186))
    
    lte_fire_burg_alarm = property(lambda self: self._make_ip(GUEST_B_OFFSET, 130))