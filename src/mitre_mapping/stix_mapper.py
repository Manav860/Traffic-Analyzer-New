"""
MITRE ATT&CK STIX 2.1 Mapper.
Translates raw dataset attack labels into standard MITRE tactics, techniques, and kill chain stages.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent.parent.parent / "mitre_mapping" / "label_to_tactic.yaml"

class MitreMapper:
    """Provides lookup and semantic enrichment for MITRE ATT&CK tactics."""

    def __init__(self, mapping_path: Optional[str] = None):
        self.mapping_path = Path(mapping_path) if mapping_path else DEFAULT_MAPPING_PATH
        self.data = self._load_mapping()
        self.mappings: Dict[str, Dict[str, Any]] = self.data.get("mappings", {})
        self.kill_chain: List[Dict[str, Any]] = self.data.get("kill_chain_order", [])

    def _load_mapping(self) -> Dict[str, Any]:
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"MITRE mapping file not found at: {self.mapping_path}")
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_tactic_for_label(self, label: str) -> Dict[str, Any]:
        """
        Lookup MITRE tactic and technique metadata for a given dataset label.
        Defaults to Benign / Unknown if not found.
        """
        # Exact match
        if label in self.mappings:
            return self.mappings[label]
        
        # Substring / fuzzy match for variants like 'DoS-Hulk', 'PortScan-SYN', etc.
        for mapped_label, info in self.mappings.items():
            if mapped_label.lower() in label.lower() or label.lower() in mapped_label.lower():
                return info

        # Keyword-based semantic fallback for Web Attacks, SQLi, XSS, Brute Force, Scan, etc.
        low_label = label.lower()
        if "brute force" in low_label or "patator" in low_label:
            return self.mappings.get("FTP-Patator", {
                "tactic_id": "TA0001",
                "tactic_name": "Initial Access",
                "technique_id": "T1110",
                "technique_name": "Brute Force",
                "kill_chain_stage": 2,
                "is_malicious": True,
                "description": f"Credential brute-force attack: {label}"
            })
        if "sql" in low_label or "injection" in low_label:
            return {
                "tactic_id": "TA0001",
                "tactic_name": "Initial Access",
                "technique_id": "T1190",
                "technique_name": "Exploit Public-Facing Application (SQLi)",
                "kill_chain_stage": 2,
                "is_malicious": True,
                "description": f"Web application exploit attempt: {label}"
            }
        if "xss" in low_label:
            return {
                "tactic_id": "TA0001",
                "tactic_name": "Initial Access",
                "technique_id": "T1189",
                "technique_name": "Drive-by Compromise (XSS)",
                "kill_chain_stage": 2,
                "is_malicious": True,
                "description": f"Cross-site scripting attempt: {label}"
            }
        if "scan" in low_label:
            return self.mappings.get("PortScan", {
                "tactic_id": "TA0043",
                "tactic_name": "Reconnaissance",
                "technique_id": "T1595",
                "technique_name": "Active Scanning",
                "kill_chain_stage": 1,
                "is_malicious": True,
                "description": f"Network scanning probe: {label}"
            })
        if "bot" in low_label:
            return self.mappings.get("Bot", {
                "tactic_id": "TA0011",
                "tactic_name": "Command and Control",
                "technique_id": "T1071",
                "technique_name": "Application Layer Protocol",
                "kill_chain_stage": 4,
                "is_malicious": True,
                "description": f"Botnet C2 communication: {label}"
            })
        if "dos" in low_label or "ddos" in low_label:
            return self.mappings.get("DDoS", {
                "tactic_id": "TA0040",
                "tactic_name": "Impact",
                "technique_id": "T1498",
                "technique_name": "Network Denial of Service",
                "kill_chain_stage": 5,
                "is_malicious": True,
                "description": f"Denial of service attack: {label}"
            })

        # Unknown / Unlabeled fallback
        return {
            "tactic_id": "TA_UNKNOWN",
            "tactic_name": "Uncategorized Attack" if label.lower() != "benign" else "Benign",
            "technique_id": "T_UNKNOWN",
            "technique_name": label,
            "kill_chain_stage": 1 if label.lower() != "benign" else 0,
            "is_malicious": label.lower() != "benign",
            "description": f"Observed network traffic labeled as '{label}'."
        }

    def get_stage_name_from_index(self, index: int) -> str:
        """Return tactical stage name from progression index."""
        stages = [
            "Benign",
            "Reconnaissance",
            "Initial Access",
            "Lateral Movement",
            "Command and Control",
            "Exfiltration",
            "Impact"
        ]
        if 0 <= index < len(stages):
            return stages[index]
        return "Unknown Stage"

    def get_all_tactics(self) -> List[str]:
        """Return unique list of mapped MITRE tactics."""
        return [
            "Benign",
            "Reconnaissance",
            "Initial Access",
            "Lateral Movement",
            "Command and Control",
            "Exfiltration",
            "Impact"
        ]


if __name__ == "__main__":
    mapper = MitreMapper()
    print("Testing MITRE STIX Mapper...")
    for label in ["Benign", "PortScan", "SSH-Bruteforce", "Infiltration", "Bot"]:
        info = mapper.get_tactic_for_label(label)
        print(f"[{label}] -> Tactic: {info['tactic_name']} ({info['tactic_id']}), Technique: {info['technique_name']}")
