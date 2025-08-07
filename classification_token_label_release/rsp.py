def calculate_expected_score(prob_r, prob_s, prob_p, sequence):
    # Define win/loss outcomes: (NiuNiu's move, NiuMei's move) -> score
    outcomes = {
        ('R', 'S'): 1,  # Rock beats Scissors
        ('R', 'P'): -1, # Paper beats Rock
        ('R', 'R'): 0,  # Tie
        ('S', 'P'): 1,  # Scissors beats Paper
        ('S', 'R'): -1, # Rock beats Scissors
        ('S', 'S'): 0,  # Tie
        ('P', 'R'): 1,  # Paper beats Rock
        ('P', 'S'): -1, # Scissors beats Paper
        ('P', 'P'): 0   # Tie
    }
    
    expected_score = 0.0
    # For each move in NiuMei's sequence
    for move in sequence: 
        # Calculate expected score for this move
        score_r = prob_r * outcomes[('R', move)]  # NiuNiu plays Rock
        score_s = prob_s * outcomes[('S', move)]  # NiuNiu plays Scissors
        score_p = prob_p * outcomes[('P', move)]  # NiuNiu plays Paper
        expected_score += score_r + score_s + score_p
    
    return expected_score

def main():
    # Read number of test cases
    t = int(input())
    
    # Process each test case
    for _ in range(t):
        # Read probabilities
        prob_r, prob_s, prob_p = map(float, input().split())
        # Read NiuMei's sequence
        sequence = input().strip()
        # Calculate and output expected score
        result = calculate_expected_score(prob_r, prob_s, prob_p, sequence)
        print(f"{result:.6f}")

if __name__ == "__main__":
    main()