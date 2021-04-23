import os

def execute_command(cmd):
   print(cmd)
   print()
   # os.system(cmd)
   return

pwd_str = os.getcwd()
if pwd_str[-1]!="/":
   pwd_str += "/"
sat_dir = pwd_str+"test/sat/"
unsat_dir = pwd_str+"test/unsat/"

# os.chdir(sat_dir)
for cnf_file in os.listdir(sat_dir):
   if cnf_file.endswith(".cnf"):
      cnf_file_path = sat_dir+cnf_file
      cmd = "time MODE=1 ./preprocessor " + cnf_file_path
      execute_command(cmd)
      cmd = "time MODE=0 ./preprocessor " + cnf_file_path
      execute_command(cmd)
      
for cnf_file in os.listdir(unsat_dir):
   if cnf_file.endswith(".cnf"):
      cnf_file_path = unsat_dir+cnf_file
      cmd = "time MODE=1 ./preprocessor " + cnf_file_path
      execute_command(cmd)
      cmd = "time MODE=0 ./preprocessor " + cnf_file_path
      execute_command(cmd)
      
