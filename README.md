# 🛡️ PC Traffic & IP Analyzer

> A Python-based Desktop Security & Network Traffic Analyzer designed to inspect connected IP addresses, analyze incoming/outgoing network traffic, and detect malicious IP addresses in real-time.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [Usage Guide](#-usage-guide)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License & Contact](#-license--contact)

---

## 📖 Overview

**PC Traffic Analyzer** is an interactive desktop security application that allows users to monitor network connections and analyze active IP addresses on their system. Built with **Python** and **Tkinter**, it scans active sockets, parses domain/IP links, and alerts users if a malicious or suspicious IP address attempts to interact with their system.

---

## ✨ Key Features

* 🚨 **Malicious IP Detection:** Scans connected IP addresses and flags suspicious or blacklisted IPs instantly.
* 🌐 **Active IP & Connection Inspector:** Displays active local and remote connections, listening ports, and destination links.
* 📊 **Traffic Data Analysis:** Uses numerical and statistical processing to inspect network metrics and packet frequency.
* 🖥️ **Desktop GUI Interface:** Intuitive user interface built using Tkinter for effortless real-time monitoring.
* 🔍 **Pattern Matching & Parsing:** Regex-powered string parsing for quick domain and IP extraction.

---

## 🛠️ Tech Stack & Dependencies

| Category | Library / Module | Description |
| :--- | :--- | :--- |
| **GUI Framework** | `tkinter` | Native Desktop User Interface |
| **Networking** | `socket` | Socket creation, IP binding, and low-level network operations |
| **Data & Analysis** | `scipy` / `numpy` | Data processing and numerical analysis of traffic flows |
| **Pattern Matching** | `re` | Regular expressions for parsing IP addresses and URLs |
| **System Operations** | `os` / `sys` | System-level interaction and network querying |

---

## 🏗️ System Architecture

```text
[ Incoming/Outgoing Network Traffic ]
                 │
                 ▼
[ Socket Listener & Scanner (`socket`) ]
                 │
                 ▼
[ IP Parsing & Regex Matching (`re`) ]
                 │
                 ├── Check against Malicious Threat Database
                 ├── Statistical Traffic Analysis (`scipy`)
                 │
                 ▼
[ Tkinter Graphical User Interface ]
  ├── Connected IP List & Links
  └── Threat Alerts & Status Display
