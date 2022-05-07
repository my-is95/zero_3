# core.py を使用するように変更
is_simple_core = False

if is_simple_core:
    """
    下記のimport文により、外部から
    from dezero improt Variable
    のような形で読み込むことができるようになる。
    """
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable
else:
    from dezero.core import Variable
    from dezero.core import Parameter
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable

# Variableインスタンスの演算子のオーバーロードを行う関数を実行
setup_variable()