# 🚦 CrashGuard: A BMSSP based Data-Driven and Risk-Aware Routing System for Safer Urban Travel Using Spatial Analytics and Machine Learning

A **data-driven, safety-conscious route recommendation system** designed to combat rising urban road accidents by prioritizing *safety alongside efficiency*. This project leverages **spatial analytics, machine learning, graph algorithms, and real-time weather data** to recommend the *safest possible routes* in metropolitan environments such as **Bengaluru**.

---

## 📌 Problem Statement

With rapid urbanization and a surge in vehicular traffic, metropolitan cities are witnessing an alarming increase in road accidents. Traditional navigation systems primarily optimize for **shortest distance** or **minimum travel time**, offering little to no insight into:

* Accident-prone zones
* Road-level safety risks
* Dynamic environmental conditions (rain, fog, etc.)

This lack of safety-aware routing poses a serious public safety challenge.

---

## 💡 Proposed Solution

**SafeRoute Intelligence** introduces a **smart routing framework** that integrates historical accident data, spatial risk modeling, and real-time weather conditions to recommend routes that are not just fast—but *safer*.

The system dynamically balances **distance** and **risk**, allowing users to travel through routes that minimize exposure to high-accident zones and hazardous environmental conditions.

---

## 🧠 System Architecture & Methodology

### 1️⃣ Accident Hotspot Detection

* Historical accident datasets are analyzed
* **DBSCAN clustering** is used to identify dense, accident-prone regions
* These clusters represent high-risk zones

### 2️⃣ Risk Propagation Modeling

* A **BallTree-based spatial model** propagates risk from hotspots
* Risk values are assigned to nearby road segments
* Road network extracted from **OpenStreetMap (OSM)**

### 3️⃣ Composite Road Weighting

Each road segment is assigned a **multi-dimensional weight**, combining:

* 📏 Normalized distance
* ⚠️ Risk score from accident proximity
* 🌦️ Environmental risk modifiers

### 4️⃣ Safe Route Computation

* A **Bounded Multi-Source Shortest Path (BMSSP)** algorithm is used
* Computes optimal paths under distance and safety constraints
* Supports safety-priority routing preferences

### 5️⃣ Real-Time Weather Integration

* Weather data fetched using **OpenWeatherMap API**
* Conditions like **rain and fog** dynamically adjust risk weightings
* Ensures routes adapt to evolving environmental hazards

---

## 🌐 Web Application

The entire system is deployed as an **interactive Streamlit web application**, enabling:

* 🗺️ Real-time route visualization
* ⚙️ User-defined safety preferences
* 🔄 Dynamic route updates based on live weather
* 📊 Transparent risk-aware routing decisions

---

## 🛠️ Tech Stack

* **Python** – Core implementation
* **Scikit-learn** – DBSCAN clustering
* **BallTree** – Spatial risk propagation
* **OSMnx / OpenStreetMap** – Road network extraction
* **NetworkX** – Graph-based routing
* **OpenWeatherMap API** – Real-time weather data
* **Streamlit** – Interactive web interface

---

## 📊 Experimental Evaluation

* Evaluated using **real-world Bengaluru traffic and accident datasets**
* Demonstrates effective identification of accident-prone zones
* Produces routes that significantly reduce exposure to high-risk areas
* Confirms the feasibility of safety-first navigation in dense urban settings

---

## 🎯 Key Contributions

* Introduces a **safety-aware routing paradigm**
* Integrates spatial ML, graph theory, and real-time data
* Provides an interpretable and customizable routing experience
* Bridges the gap between navigation efficiency and road safety

---

## 🔮 Future Enhancements

* Live traffic congestion integration
* User feedback-based risk learning
* Expansion to multi-city deployment
* Mobile application support
* Accident severity-aware modeling

---

## 📍 Use Cases

* Urban commuters prioritizing safety
* Emergency response route planning
* City traffic management authorities
* Academic research in smart transportation

---

## 📄 License

This project is intended for academic, research, and demonstration purposes.

---

## 🚀 Conclusion

**SafeRoute Intelligence** demonstrates that intelligent, safety-aware navigation is both feasible and impactful. By combining machine learning, spatial analytics, and real-time environmental data, this framework takes a decisive step toward **safer urban mobility**.

---

**Navigate Smart. Navigate Safe. 🚦**
