const fetch = require("node-fetch");

fetch('http://192.168.1.68:8000')
  .then((response) => {
    return response;
  })
  .then((data) => {
    console.log(data);
  });