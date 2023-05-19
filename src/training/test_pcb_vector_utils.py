"""Unit tests for pcb_vector_utils module"""
import pcb_vector_utils

def test_calculate_resultant_vector():
    """Tests rare condition of having both inputs 0.0 which if unchecked, will \
        return an angle value of nan."""
    euclidean_dist, angle = pcb_vector_utils.calculate_resultant_vector(0,0)
    assert euclidean_dist == 0.0
    assert angle == 0.0
