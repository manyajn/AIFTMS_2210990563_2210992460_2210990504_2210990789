from confluent_kafka import Consumer, Producer
from sklearn.ensemble import IsolationForest
import joblib, json

model: IsolationForest = joblib.load('models/isolation_forest_v3.pkl')
ANOMALY_THRESHOLD = -0.15

consumer = Consumer({
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'fraud-scorer',
    'auto.offset.reset': 'latest'
})
producer = Producer({'bootstrap.servers': 'kafka:9092'})
consumer.subscribe(['finance.ap.transactions'])
