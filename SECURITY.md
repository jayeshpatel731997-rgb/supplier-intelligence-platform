# Security & Data Privacy Architecture
## Agentic AI Supplier Risk Intelligence System

**Version:** 1.0
**Last Updated:** February 2026
**Status:** Research Prototype → Pre-Production

---

## 1. Data Classification

All data in this system falls into one of four categories:

### PUBLIC (No restriction)
- News articles from RSS feeds and NewsAPI
- USGS earthquake data
- Federal Register tariff/trade publications
- Published academic research
- General industry statistics

**Handling:** Can be sent to external APIs, cached freely, displayed openly.

### CONFIDENTIAL (Client proprietary)
- Supplier names, locations, contact information
- Spend data, contract values, pricing
- On-time delivery rates, defect rates
- Supplier network topology (who supplies whom)
- Internal risk assessments and scores

**Handling:**
- NEVER sent to external LLM APIs (Claude, OpenAI, etc.)
- Stored locally or in client-controlled infrastructure
- Encrypted at rest (AES-256) and in transit (TLS 1.3)
- Access restricted to authenticated users only
- Deleted upon client request within 30 days

### SENSITIVE (Regulated)
- Defense/government supply chain data (ITAR/EAR)
- Financial data subject to SOX compliance
- Personal data of supplier contacts (GDPR/CCPA)

**Handling:** Not accepted in current version. Enterprise feature with dedicated compliance.

### SYSTEM (Internal)
- Model parameters, weights, calibration data
- API keys and authentication tokens
- Application logs and error traces

**Handling:** Stored in environment variables, never committed to version control.

---

## 2. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT BROWSER                            │
│                  (Streamlit UI)                              │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTPS/TLS 1.3
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 APPLICATION SERVER                            │
│              (Streamlit + Python)                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Mathematical  │  │   Sentinel   │  │   Client Data    │  │
│  │ Models        │  │   Agent      │  │   Store          │  │
│  │ (Local CPU)   │  │              │  │   (Local/Cloud)  │  │
│  │               │  │  PUBLIC data │  │                  │  │
│  │ • SIR         │  │  only sent   │  │  CONFIDENTIAL    │  │
│  │ • Bayesian    │  │  to LLM API  │  │  data stays      │  │
│  │ • Monte Carlo │  │       │      │  │  here             │  │
│  │ • Graph       │  │       ▼      │  │                  │  │
│  └──────────────┘  │  Claude API   │  │  • Supplier DB   │  │
│                    │  (news only)  │  │  • Risk scores   │  │
│                    └──────────────┘  │  • Network graph  │  │
│                                      └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘

KEY PRINCIPLE: Confidential supplier data NEVER leaves the
application server. Only public news text is sent to Claude API
for event classification.
```

---

## 3. What Goes to External APIs (and What Doesn't)

### Sentinel Agent → Claude API
**SENT:**
- News article title (public)
- News article summary text (public)
- Source name (public)
- Publication date (public)

**NEVER SENT:**
- Supplier names
- Spend amounts
- Delivery metrics
- Network topology
- Client company name
- Any data from the CONFIDENTIAL category

### Mathematical Models (SIR, Bayesian, Monte Carlo, Graph)
**All computation happens locally.** Zero external API calls.
Client data stays on the application server at all times.

---

## 4. Authentication & Access Control

### Current (Research Prototype)
- Single-user, local deployment
- No authentication required (localhost only)

### Phase 2 (Consulting/Multi-user)
- Token-based API authentication
- Session-based web authentication (Streamlit auth or OAuth)
- Role-based access:
  - ADMIN: Full access, manage users, configure system
  - ANALYST: View all data, run simulations, generate reports
  - VIEWER: Read-only access to dashboards and reports

### Phase 3 (SaaS/Enterprise)
- SSO integration (SAML 2.0 / OAuth 2.0)
- Multi-factor authentication (MFA)
- API key management with scoping and rotation
- Audit logging of all data access

---

## 5. Data Retention & Deletion

| Data Type | Default Retention | Client Can Request |
|-----------|------------------|-------------------|
| News articles (public) | 90 days | N/A |
| Sentinel events | 90 days | Deletion |
| Supplier data | Duration of engagement | Full deletion within 30 days |
| Simulation results | 90 days | Deletion |
| Risk reports | 1 year | Deletion |
| Application logs | 30 days | N/A |
| API request logs | 7 days (no payloads) | N/A |

**Right to Deletion:** Clients can request complete deletion of all their
proprietary data. Upon request, data is purged from all storage within 30 days
and confirmation is provided in writing.

---

## 6. Infrastructure Security

### Current Deployment (Streamlit Cloud)
- HTTPS enforced (TLS 1.3)
- Streamlit Cloud handles DDoS protection
- No persistent storage of confidential data (session-based)
- API keys stored in Streamlit Secrets (encrypted)

### Production Deployment (Future)
- AWS/GCP with VPC isolation
- Database encryption at rest (AES-256)
- Regular automated backups (encrypted)
- Network security groups / firewall rules
- Container-based deployment (Docker)
- Infrastructure as Code (Terraform)

---

## 7. Incident Response

### If a Data Breach Occurs:
1. **Contain** — Isolate affected systems within 1 hour
2. **Assess** — Determine scope and affected data
3. **Notify** — Inform affected clients within 72 hours (GDPR requirement)
4. **Remediate** — Fix vulnerability, restore from backup
5. **Report** — Post-incident report within 30 days

### Security Contact
Report vulnerabilities to: security@[yourdomain].com

---

## 8. Compliance Roadmap

| Standard | Status | Target Date |
|----------|--------|-------------|
| Basic HTTPS + Auth | ✅ Implemented | Feb 2026 |
| Data Classification | ✅ Documented | Feb 2026 |
| API Data Isolation | ✅ Implemented | Feb 2026 |
| Privacy Policy | 📝 In Progress | Mar 2026 |
| Terms of Service | 📝 In Progress | Mar 2026 |
| SOC 2 Type I | 🔜 Planned | Q4 2026 |
| GDPR Compliance | 🔜 Planned | Q1 2027 |
| SOC 2 Type II | 🔜 Planned | Q2 2027 |

---

## 9. Open Source & Third-Party Dependencies

| Dependency | Purpose | Data Access |
|-----------|---------|-------------|
| Anthropic Claude API | News classification only | Public text only |
| Streamlit | Web interface | Renders UI, no data storage |
| NetworkX | Graph computation | Local only |
| NumPy/SciPy | Mathematical models | Local only |
| Plotly | Visualization | Client-side rendering |
| NewsAPI | News feed | Sends search queries only |
| USGS API | Earthquake data | Public data only |
| feedparser | RSS parsing | Public feeds only |

**No third-party dependency has access to confidential supplier data.**

---

*This document is part of the Agentic AI Supplier Risk Intelligence System
research project. For security inquiries, contact the development team.*
