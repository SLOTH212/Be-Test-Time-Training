from dynamic_ttt.mechanisms.runtime import reverse
def test_exact():
 s=['L0','OFF','ALL','L30'];assert reverse(s)==s[::-1];assert reverse(reverse(s))==s
