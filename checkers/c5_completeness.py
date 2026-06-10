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
        
        self.coverage_threshold = 0.8
    
    def check(self, scenario, explanation, context=None):
        """
        Check completeness using structured contributing_factors if available.
        Also accesses primary_cause for additional context.
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
        
        # Adjust threshold based on context
        if context:
            mechanism_failed = context.has_violation('C3')
            if mechanism_failed:
                self.coverage_threshold = 0.9
            else:
                self.coverage_threshold = 0.8
        
        # Use structured contributing factors if available
        if contributing_factors:
            mentioned = []
            for factor in required_factors:
                factor_lower = factor.lower()
                # Check against structured factors
                if any(factor_lower in cf.lower() or cf.lower() in factor_lower for cf in contributing_factors):
                    mentioned.append(factor)
                # Also check against primary_cause
                elif primary_cause and factor_lower in primary_cause.lower():
                    mentioned.append(factor)
                # Fallback to text search
                elif factor_lower in explanation_text.lower():
                    mentioned.append(factor)
            used_structured = True
        else:
            # Fallback to text search
            mentioned = [f for f in required_factors if f.lower() in explanation_text.lower()]
            used_structured = False
        
        missing = [f for f in required_factors if f not in mentioned]
        
        # Identify core factors
        core_factors = self._get_core_factors(required_factors, scenario)
        core_missing = [f for f in core_factors if f in missing]
        
        coverage = len(mentioned) / len(required_factors)
        passed = (coverage >= self.coverage_threshold) or (len(core_missing) == 0)
        
        confidence = coverage
        if core_missing:
            confidence *= 0.7
        
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
                'used_structured': used_structured,
                'primary_cause_used': bool(primary_cause and used_structured)
            }
        }
    
    def _get_core_factors(self, factors, scenario):
        """Identify which factors are core/essential"""
        category = scenario.get('category', '')
        
        category_core = {
            'Weather': ['weather_event', 'road_condition', 'primary_cause'],
            'Traffic Accident': ['driver_action', 'primary_cause', 'collision_type'],
            'Road Maintenance': ['maintenance_activity', 'safety_failure', 'hazard'],
            'Public Event': ['event_type', 'capacity_issue', 'traffic_impact']
        }.get(category, [])
        
        core = []
        for f in factors:
            f_lower = f.lower()
            for core_pattern in category_core:
                if core_pattern in f_lower:
                    core.append(f)
                    break
        
        if not core and len(factors) >= 2:
            core = factors[:2]
        
        return core
    
    def _get_template_factors(self, category):
        """Get required factors from category template"""
        for template in self.templates:
            if template.get('category') == category:
                return template.get('required', [])
        
        domain_factors = {
            'Weather': ['weather_event', 'road_condition', 'driver_action'],
            'Traffic Accident': ['primary_cause', 'contributing_factor', 'outcome'],
            'Road Maintenance': ['maintenance_activity', 'safety_failure', 'resulting_hazard'],
            'Public Event': ['event_type', 'capacity_issue', 'traffic_impact'],
            'Healthcare': ['primary_condition', 'contributing_factors', 'outcome'],
            'Finance': ['trigger_event', 'market_condition', 'result']
        }
        
        return domain_factors.get(category, [])