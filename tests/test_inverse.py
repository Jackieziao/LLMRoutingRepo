from llm_route_opt.inverse import inverse_example


def test_discrete_inverse_example_is_rationalized() -> None:
    result = inverse_example()
    assert result.pairwise_accuracy == 1
    assert abs(sum(result.weights.values()) - 1) < 1e-12
    assert result.weights["quality"] > result.weights["economy"]
