# drugReview
pip install -r requirements.txt
docker build -t drug-review-api:v1 .
docker run -p 6000:6000 drug-review-api:v1


docker login --username=avijitindocker
docker tag drug-review-api:v1 avijitindocker/drug-review-api:v1
docker push avijitindocker/drug-review-api:v1

docker pull avijitindocker/drug-review-api:v1

External Access Configuration
Reserve a Static External IP for your VM in the GCP Console under “VPC network” > “External IP addresses”.
Configure Firewall Rules:
Go to “VPC network” > “Firewall”.
Create a new firewall rule:
Name: Streamlit-app-rule
Targets: All instances in the network
Source IP ranges: 0.0.0.0/0 (or restrict as needed)
Specified protocols and ports: tcp:8501 (Streamlit's default port)
