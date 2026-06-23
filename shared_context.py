# shared_context.py file
class CheckerContext:
    """
    Shared context object that passes results between checkers.
    Each checker can read previous results and add its own.
    """
    
    def __init__(self):
        self.results = {}
        self.violations = []
        self.notes = []
    
    def add_result(self, checker_id, result):
        """Store result from a checker"""
        self.results[checker_id] = result
        if not result['passed']:
            self.violations.append({
                'checker': checker_id,
                'reason': result['reason'],
                'confidence': result.get('confidence', 0)
            })
    
    def get_result(self, checker_id):
        """Get result from a previous checker"""
        return self.results.get(checker_id)
    
    def has_violation(self, checker_id):
        """Check if a specific checker failed"""
        result = self.get_result(checker_id)
        return result is not None and not result['passed']
    
    def add_note(self, note):
        """Add a note for debugging/explainability"""
        self.notes.append(note)
    
    def get_summary(self):
        """Get overall assessment"""
        return {
            'violations': self.violations,
            'notes': self.notes,
            'all_passed': len(self.violations) == 0,
            'severity': len(self.violations)
        }