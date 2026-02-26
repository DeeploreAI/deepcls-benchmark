"""
Test suite for auto_pad function
Tests various kernel sizes and strides to verify correct padding calculation
"""

import math
from typing import Dict, List
from models.common import auto_pad


class AutoPadTester:
    """Test class for auto_pad function validation"""
    
    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        self.test_results = []
    
    def calculate_output_size(self, k: int, s: int, d: int = 1) -> Dict:
        """
        Calculate expected vs actual output size for given parameters
        
        Args:
            k: kernel size
            s: stride
            d: dilation
            
        Returns:
            Dictionary containing server results
        """
        # Get padding from auto_pad function
        padding = auto_pad(k, s, d)
        
        # Calculate effective kernel size
        k_eff = d * (k - 1) + 1
        
        # Calculate actual output using convolution formula
        actual_output = math.floor((self.input_size + 2 * padding - k_eff) / s) + 1
        
        # Determine expected output based on stride
        if s == 1:
            expected_output = self.input_size  # Same shape
        elif s == 2:
            expected_output = self.input_size // 2  # Half shape
        else:
            expected_output = self.input_size // s  # Approximate downsampling
        
        return {
            'k': k, 's': s, 'd': d,
            'padding': padding,
            'k_eff': k_eff,
            'expected': expected_output,
            'actual': actual_output,
            'passed': actual_output == expected_output,
            'error': abs(actual_output - expected_output)
        }
    
    def run_comprehensive_tests(self) -> bool:
        """
        Run comprehensive server cases covering all scenarios
        
        Returns:
            True if all tests pass, False otherwise
        """
        # Define server cases covering all branches
        test_cases = [
            # Stride=1 cases (same shape)
            (1, 1, 1, "1x1 pointwise convolution"),
            (3, 1, 1, "3x3 standard convolution"),
            (5, 1, 1, "5x5 large convolution"),
            
            # Stride=2, odd kernel cases (half shape)
            (3, 2, 1, "3x3 stride-2 downsampling"),
            (5, 2, 1, "5x5 stride-2 downsampling"),
            
            # Stride=2, even kernel cases (half shape)
            (2, 2, 1, "2x2 standard pooling"),
            (4, 2, 1, "4x4 large pooling"),
            
            # Higher stride cases
            (3, 4, 1, "3x3 stride-4 aggressive downsampling"),
            
            # Dilation cases
            (3, 1, 2, "3x3 dilated convolution"),
            (3, 2, 2, "3x3 dilated stride-2"),
        ]
        
        print("=== Auto Pad Function Comprehensive Test ===")
        print(f"Input size: {self.input_size}")
        print("-" * 70)
        
        all_passed = True
        
        for i, (k, s, d, description) in enumerate(test_cases, 1):
            result = self.calculate_output_size(k, s, d)
            self.test_results.append(result)
            
            # Format output
            status = "PASS" if result['passed'] else "FAIL"
            status_icon = "✅" if result['passed'] else "❌"
            
            print(f"Test {i:2d}: {description}")
            print(f"         k={k}, s={s}, d={d} → padding={result['padding']}")
            print(f"         Expected: {result['expected']}, Actual: {result['actual']} [{status_icon} {status}]")
            
            if not result['passed']:
                print(f"         Error: {result['error']} pixels")
                all_passed = False
            
            print()
        
        # Summary
        passed_count = sum(1 for r in self.test_results if r['passed'])
        total_count = len(self.test_results)
        
        print("-" * 70)
        print(f"Summary: {passed_count}/{total_count} tests passed")
        
        if all_passed:
            print("🎉 All tests PASSED!")
        else:
            print("⚠️  Some tests FAILED!")
        
        return all_passed
    
    def analyze_padding_patterns(self) -> None:
        """Analyze and display padding patterns for different parameter combinations"""
        print("\n=== Padding Pattern Analysis ===")
        
        # Test various combinations to understand patterns
        analysis_cases = [
            (1, 1), (3, 1), (5, 1),  # stride=1 cases
            (3, 2), (5, 2),          # stride=2, odd kernel
            (2, 2), (4, 2), (6, 2),  # stride=2, even kernel
            (3, 3), (3, 4),          # higher strides
        ]
        
        for k, s in analysis_cases:
            padding = auto_pad(k, s, 1)
            k_eff = k  # d=1
            
            # Classify the case
            if k_eff % 2 == 1:
                case_type = "Odd kernel"
            else:
                case_type = "Even kernel"
            
            print(f"k={k}, s={s}: padding={padding:2d} ({case_type})")
            
            # Show output sizes for different input sizes
            for input_sz in [224, 112, 56]:
                output = math.floor((input_sz + 2 * padding - k_eff) / s) + 1
                ratio = output / input_sz
                print(f"  {input_sz:3d} → {output:3d} (ratio: {ratio:.3f})")
            print()
    
    def get_failed_cases(self) -> List[Dict]:
        """Return list of failed server cases for debugging"""
        return [result for result in self.test_results if not result['passed']]


def main():
    """Main function to run all tests"""
    tester = AutoPadTester(input_size=224)
    
    # Run comprehensive tests
    success = tester.run_comprehensive_tests()
    
    # Analyze padding patterns
    tester.analyze_padding_patterns()
    
    # Show failed cases if any
    failed_cases = tester.get_failed_cases()
    if failed_cases:
        print("\n=== Failed Cases Analysis ===")
        for case in failed_cases:
            print(f"k={case['k']}, s={case['s']}, d={case['d']}")
            print(f"  Expected: {case['expected']}, Got: {case['actual']}")
            print(f"  Padding: {case['padding']}, k_eff: {case['k_eff']}")
            print()
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 