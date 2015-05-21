def git_version():
    from subprocess import Popen, PIPE
    gitproc = Popen(['git', 'rev-parse','--short','HEAD'], stdout = PIPE)
    (stdout, _) = gitproc.communicate()
    return stdout.strip()
