# checkers/c5_completeness.py
import re
import json
from pathlib import Path

class C5CompletenessChecker:
    def __init__(self, kb_path='data/completeness_templates.json'):
        self.name = "C₅ Completeness"
        
        kb_path = Path(__file__).parent.parent / kb_path
        if kb_path.exists():
            with open(kb_path, 'r', encoding='utf-8') as f:
                self.templates = json.load(f)['templates']
        else:
            self.templates = []
        
        self.coverage_threshold = 0.5  # Lowered from 0.8 for better recall
    
    def check(self, scenario, explanation, context=None):
        """
        Check completeness using structured contributing_factors if available.
        """
        # Extract structured data
        if isinstance(explanation, dict):
            structured = explanation.get('structured_output', {})
            contributing_factors = structured.get('contributing_factors', [])
            primary_cause = structured.get('primary_cause', '')
            explanation_text = explanation.get('explanation', '')
        else:
            structured = {}
            contributing_factors = []
            primary_cause = ''
            explanation_text = explanation
        
        required_factors = scenario.get('minimal_sufficient_set', [])
        
        if not required_factors:
            category = scenario.get('category', '')
            required_factors = self._get_template_factors(category)
        
        if not required_factors:
            return {
                'checker': 'C5',
                'passed': True,
                'confidence': 0.5,
                'reason': 'No required factors specified',
                'details': {'warning': 'Missing minimal_sufficient_set in scenario'}
            }
        
        print(f"\n🔍 C5 DEBUG for {scenario.get('id', 'unknown')}:")
        print(f"   Required factors: {required_factors}")
        print(f"   Mentioned (structured): {contributing_factors}")
        print(f"   Mentioned (text): {explanation_text[:200]}...")
        
        # Adjust threshold based on context
        if context:
            mechanism_failed = context.has_violation('C3')
            if mechanism_failed:
                self.coverage_threshold = 0.6  # Less strict when mechanism failed
            else:
                self.coverage_threshold = 0.5
        
        # Check which factors are mentioned
        mentioned = []
        missing = []
        partial_matches = []
        
        for factor in required_factors:
            is_mentioned, match_type = self._factor_mentioned(factor, explanation_text, contributing_factors, primary_cause)
            if is_mentioned:
                mentioned.append(factor)
                if match_type == 'partial':
                    partial_matches.append(factor)
            else:
                missing.append(factor)
        
        # CALCULATE coverage AND passed BEFORE printing them
        coverage = len(mentioned) / len(required_factors) if required_factors else 1.0
        
        # Identify core factors
        core_factors = self._get_core_factors(required_factors, scenario)
        core_missing = [f for f in core_factors if f in missing]
        
        # More lenient pass logic
        passed = (coverage >= self.coverage_threshold) or (len(core_missing) <= 1)
        
        # Calculate confidence
        confidence = coverage
        if core_missing:
            confidence *= 0.8  # Less penalty for core missing
        
        # Boost confidence if using structured data
        if contributing_factors:
            confidence = min(1.0, confidence + 0.1)
        
        # NOW print the debug info (after variables are defined)
        print(f"   Final mentioned: {mentioned}")
        print(f"   Final missing: {missing}")
        print(f"   Coverage: {coverage:.2%}")
        print(f"   Passed: {passed}")
        
        return {
            'checker': 'C5',
            'passed': passed,
            'confidence': round(confidence, 3),
            'reason': f'Coverage: {coverage:.1%} ({len(mentioned)}/{len(required_factors)})',
            'details': {
                'required': required_factors,
                'mentioned': mentioned,
                'missing': missing,
                'core_factors': core_factors,
                'core_missing': core_missing,
                'coverage': coverage,
                'threshold': self.coverage_threshold,
                'used_structured': bool(contributing_factors),
                'partial_matches': partial_matches
            }
        }
    
    def _factor_mentioned(self, factor, explanation_text, contributing_factors, primary_cause):
        """
        Check if a factor is mentioned with synonym matching.
        Returns (is_mentioned, match_type) where match_type is 'exact', 'synonym', 'partial', or 'none'
        """
        factor_lower = factor.lower()
        combined_text = f"{explanation_text} {' '.join(contributing_factors)} {primary_cause}".lower()
        
        # Direct exact match
        if factor_lower in combined_text:
            return True, 'exact'
        
        # Expanded synonym mapping
        synonym_mappings = {
            'primary_cause': ['cause', 'root cause', 'due to', 'because of', 'triggered by', 'main cause'],
            'contributing_factor': ['contributed to', 'played a role', 'factor', 'influenced', 'also caused'],
            'outcome': ['result', 'led to', 'resulted in', 'caused', 'produced'],
            'weather_event': ['rain', 'fog', 'snow', 'storm', 'wind', 'ice', 'hail', 'drizzle'],
            'road_condition': ['wet', 'slick', 'icy', 'dry', 'flooded', 'slippery', 'ponding'],
            'driver_action': ['braked', 'swerved', 'accelerated', 'changed lanes', 'turned', 'steered'],
            'maintenance_activity': ['repair', 'patching', 'resurfacing', 'inspection', 'paving'],
            'capacity_issue': ['overcrowding', 'saturation', 'bottleneck', 'overflow', 'congestion', 'queue'],
            'human_error': ['mistake', 'error', 'oversight', 'negligence', 'inattention'],
            'vehicle_factor': ['blowout', 'tire failure', 'brake fade', 'mechanical', 'malfunction'],
        }
        
        # Check synonyms
        for key, synonyms in synonym_mappings.items():
            if key in factor_lower or factor_lower in key:
                for syn in synonyms:
                    if syn in combined_text:
                        return True, 'synonym'
        
        # Check for word stems (partial matches)
        factor_words = factor_lower.split('_')
        for word in factor_words:
            if len(word) > 3 and word in combined_text:
                return True, 'partial'
        
        return False, 'none'
    
    def _get_core_factors(self, factors, scenario):
        """Identify which factors are core/essential (more aggressive)"""
        category = scenario.get('category', '')
        
        # Always core regardless of category
        always_core = ['primary_cause', 'mechanism', 'main_factor']
        
        # Category-specific core
        category_core = {
            'Weather': ['weather_event', 'road_condition', 'driver_action'],
            'Traffic Accident': ['driver_action', 'primary_cause', 'vehicle_factor'],
            'Road Maintenance': ['maintenance_activity', 'safety_failure', 'hazard'],
            'Public Event': ['event_type', 'capacity_issue', 'traffic_impact'],
            'Healthcare': ['primary_condition', 'intervention', 'human_error'],
            'Finance': ['trigger_event', 'market_condition', 'outcome']
        }.get(category, [])
        
        core = []
        for f in factors:
            f_lower = f.lower()
            if any(core_word in f_lower for core_word in always_core + category_core):
                core.append(f)
        
        # If still no core factors, take first 1 (more aggressive)
        if not core and factors:
            core = [factors[0]]
        
        return core
    
    # In c5_completeness.py - add domain-specific required factors

def _get_template_factors(self, category):
    """Get required factors from category template."""
    domain_factors = {
        'Weather': ['weather_event', 'road_condition', 'driver_action', 'outcome'],
        'Traffic Accident': ['primary_cause', 'contributing_factor', 'vehicle_factor', 'outcome'],
        'Road Maintenance': ['maintenance_activity', 'safety_failure', 'resulting_hazard', 'outcome'],
        'Public Event': ['event_type', 'capacity_issue', 'traffic_impact', 'outcome'],
        'Healthcare': ['primary_condition', 'intervention', 'contributing_factors', 'outcome'],
        'Finance': ['trigger_event', 'market_condition', 'participants', 'result']
    }
    
    return domain_factors.get(category, ['primary_cause', 'contributing_factor', 'outcome'])