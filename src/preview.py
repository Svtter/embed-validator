# preview input data
import json

import numpy as np


class Previewer:
  """tool to preview the content of test data"""

  def preview(self, data_path: str):
    """preview the content of test data"""
    with open(data_path, "r") as f:
      data = json.load(f)
    return data

  def cmd(self):
    data = np.array(self.preview("./imgdata.json"))
    print(data.reshape(32, 1, 32, 192).shape)


if __name__ == "__main__":
  previewer = Previewer()
  previewer.cmd()
