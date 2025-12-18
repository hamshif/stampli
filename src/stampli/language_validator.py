from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, TypedDict

class IValidationRule(ABC):
    """Interface for character validation rules."""
    
    @abstractmethod
    def can_follow(self, next_char: str) -> bool:
        """Checks if the next character is allowed to follow this one."""
        pass

    @abstractmethod
    def can_be_final(self) -> bool:
        """Checks if this character is allowed to be the final character of the word."""
        pass

class CharacterRule(IValidationRule):
    """Concrete implementation of validation rules for a specific character."""
    
    def __init__(self, char: str, followers: List[str], is_final: bool):
        self.char = char
        self.followers = set(followers)
        self.is_final = is_final

    def can_follow(self, next_char: str) -> bool:
        return next_char in self.followers

    def can_be_final(self) -> bool:
        return self.is_final

class LanguageValidator:
    """
    Validator engine that checks words against a set of registered character rules.
    """
    def __init__(self):
        self._rules: Dict[str, IValidationRule] = {}

    def add_rule(self, rule: CharacterRule) -> None:
        self._rules[rule.char] = rule

    def is_valid(self, word: str) -> bool:
        """
        Checks if a word is valid according to the predefined rules.
        """
        if not word:
            return False

        for i, char in enumerate(word):
            # Check if rule exists for the character
            rule = self._rules.get(char)
            
            # If we encounter a character with no defined rules (like d, f in the example if not added),
            # we consider it invalid as we cannot verify its behavior.
            if not rule:
                return False

            # Check 'Followed by' constraint
            if i < len(word) - 1:
                next_char = word[i + 1]
                if not rule.can_follow(next_char):
                    return False
            
            # Check 'Final' constraint
            else:
                if not rule.can_be_final():
                    return False
                    
        return True

# --- Setup for the specific exercise ---

def get_exercise_validator() -> LanguageValidator:
    """Factory function to return a validator with the exercise specific rules."""
    validator = LanguageValidator()
    
    # Rules from the exercise table
    # a | [a,b,d] | true
    validator.add_rule(CharacterRule('a', ['a', 'b', 'd'], True))
    
    # b | [a,f]   | false
    validator.add_rule(CharacterRule('b', ['a', 'f'], False))
    
    # c | [a]     | true
    validator.add_rule(CharacterRule('c', ['a'], True))
    
    # Adding d and f just in case they appear as followers but don't have rules defined in the prompt explicitly
    # The prompt implies validation logic for the input string sequence. 
    # If 'd' or 'f' are encountered as PRIMARY characters in the loop, they will fail 
    # unless we add rules for them. The prompt didn't specify rules for d and f, 
    # but 'a' can be followed by 'd' and 'b' can be followed by 'f'.
    # If we strictly follow "no rules for d/f", then a string ending in 'd' or 'f' 
    # or containing 'd'/'f' followed by something else is undefined.
    # For this exercise, I will treat them as invalid if they appear as primary characters 
    # because no rule was provided for them.
    
    return validator

# --- Example Usage ---

class TestCase(TypedDict):
    word: str
    expected_result: bool
    explanation: str

if __name__ == "__main__":
    validator = get_exercise_validator()
    
    test_cases: List[TestCase] = [
        # Original Examples
        {
            "word": "ac",
            "expected_result": False,
            "explanation": "'c' is not in 'a's allowed followers [a, b, d]"
        },
        {
            "word": "ab",
            "expected_result": False,
            "explanation": "'b' cannot be final (Final=False)"
        },
        {
            "word": "aba",
            "expected_result": True,
            "explanation": "'a' follows 'b', 'b' follows 'a', 'a' is final. All valid."
        },
        
        # Single Character Edge Cases
        {
            "word": "a",
            "expected_result": True,
            "explanation": "'a' is valid final."
        },
        {
            "word": "b",
            "expected_result": False,
            "explanation": "'b' is not valid final."
        },
        {
            "word": "c",
            "expected_result": True,
            "explanation": "'c' is valid final."
        },

        # Valid Chains
        {
            "word": "aaaaa",
            "expected_result": True,
            "explanation": "'a' can always follow 'a'."
        },
        {
            "word": "ca",
            "expected_result": True,
            "explanation": "'c' -> 'a' is allowed, 'a' is final."
        },

        # Invalid Followers
        {
            "word": "cb",
            "expected_result": False,
            "explanation": "'c' can only be followed by 'a', not 'b'."
        },
        {
            "word": "bb",
            "expected_result": False,
            "explanation": "'b' cannot follow 'b' (only [a, f])."
        },
        
        # Dead Ends / Undefined Rules (d and f)
        # Note: 'a' can be followed by 'd'. So 'ad' is a sequence where we check 'a' (OK) 
        # then we check 'd'. 'd' has no rules defined in the prompt. 
        # My implementation returns False for unknown chars.
        {
            "word": "ad",
            "expected_result": False,
            "explanation": "'d' is allowed after 'a', BUT 'd' itself has no defined rules and thus cannot be validated."
        },
        
        # Empty
        {
            "word": "",
            "expected_result": False,
            "explanation": "Empty string is not valid."
        }
    ]

    print(f"{'WORD':<10} | {'EXPECTED':<10} | {'ACTUAL':<10} | {'STATUS':<10} | {'EXPLANATION'}")
    print("-" * 100)

    for case in test_cases:
        word = case["word"]
        expected = case["expected_result"]
        actual = validator.is_valid(word)
        status = "PASS" if expected == actual else "FAIL"
        
        # Color coding for terminal
        color = "\033[92m" if status == "PASS" else "\033[91m"
        reset = "\033[0m"
        
        print(f"{word:<10} | {str(expected):<10} | {str(actual):<10} | {color}{status:<10}{reset} | {case['explanation']}")
