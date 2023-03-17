from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    val: float = vals[arg]
    vals2 = list(vals)
    halfEpsilon: float = epsilon / 2
    # vals2 = vals[:, arg] + (val + halfEpsilon) + vals[arg + 1, :]
    vals2[arg] = val - halfEpsilon
    f1: float = f(*tuple(vals2))
    # vals2 = vals[:, arg] + (val - halfEpsilon) + vals[arg + 1, :]
    vals2[arg] = val + halfEpsilon
    f2: float = f(*tuple(vals2))
    return (f2 - f1) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def visit(variable: Variable, sortedList: Iterable[Variable], markedDict):
    if variable.unique_id in markedDict:
        return
    if variable.is_leaf():
        return
    if variable.is_constant():
        return
    for parent in variable.parents:
        visit(parent, sortedList, markedDict)
    sortedList.append(variable)
    markedDict[variable.unique_id] = True


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # Muss noch getestet werden
    # breakpoint()
    sortedList: Iterable[Variable] = []
    markedDict = dict()
    visit(variable, sortedList, markedDict)
    sortedList.reverse()
    return sortedList


def accumulate_derivate(variable: Variable, deriv: Any, derivatives: Iterable) -> None:
    if variable.is_leaf():
        variable.accumulate_derivative(deriv)
    else:
        if variable.unique_id in derivatives:
            derivatives[variable.unique_id] += deriv
        else:
            derivatives[variable.unique_id] = deriv


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    variables: Iterable[Variable] = topological_sort(variable)
    derivatives = dict()
    derivatives[variable.unique_id] = deriv
    for currVar in variables:
        deriv: Any = derivatives[currVar.unique_id]
        inputs: Iterable[Tuple[Variable, Any]] = currVar.chain_rule(deriv)
        for inputVar, inputDeriv in inputs:
            accumulate_derivate(inputVar, inputDeriv, derivatives)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
