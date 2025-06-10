"""pull code and update docker image"""

from fabric import Connection
from loguru import logger

code_folder = "/home/svtter/work/SWR.pytorch"
service_folder = "/home/svtter/Services/swr"


class Job:
  def __init__(self):
    self.conn = Connection("pi", user="svtter")

  def pull_code(self):
    logger.info("pull code")
    with self.conn as conn:
      conn.run("cd " + code_folder + " && git pull")

  def build_docker_image(self):
    logger.info("build docker image")
    with self.conn as conn:
      conn.run("cd " + code_folder + " &&  docker build -t svtter/swr-lite -f services/infe.Dockerfile . ")

  def restart_container(self):
    logger.info("restart container")
    with self.conn as conn:
      with conn.cd(service_folder):
        conn.run("docker compose down")
        conn.run("docker compose up -d")

  def get_logs(self):
    with self.conn as conn:
      with conn.cd(service_folder):
        return conn.run("docker compose logs -f")

  def cmd(self):
    self.pull_code()
    self.build_docker_image()
    self.restart_container()


if __name__ == "__main__":
  job = Job()
  job.cmd()
