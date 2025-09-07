import requests

# Change if running remotely
BASE = "http://127.0.0.1:8000"

print(requests.get(f"{BASE}/health").json())

rn_payload = {
  "margin_dose": 18.0,
  "v12_cc": 8.5,
  "largest_target_volume_cc": 1.2,
  "num_targets": 3,
  "piv_cc": 14.0,
  "beam_on_time_min": 42.0
}
print("RN:", requests.post(f"{BASE}/predict/rn", json=rn_payload).json())

os_payload = {
  "age": 65,
  "baseline_kps": 80,
  "sex": "male",
  "primary_histology": "NSCLC",
  "extracranial_disease": "present"
}
print("OS180:", requests.post(f"{BASE}/predict/os180", json=os_payload).json())
print("OS365:", requests.post(f"{BASE}/predict/os365", json=os_payload).json())
