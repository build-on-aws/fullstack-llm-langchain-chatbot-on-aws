/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
var request = require('request');
const axios = require('axios');

// Access the environment variable
const QUEST_URL = process.env.QUEST_URL;

module.exports = function(controller) {

    // controller.hears('sample','message,direct_message', async(bot, message) => {
    //     await bot.reply(message, 'I heard a sample message.');
    // });

    controller.on('message,direct_message,message_received', async(bot, message) => {
      const data = JSON.stringify({
        // Validation data coming from a form usually
        question: message.text
      });
      console.log(data);
      const response = await axios({
        url: QUEST_URL,
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        data: data,
      });
      console.log('Axios response:', response.data.response);
        await bot.reply(message, `${response.data.response}`);
    });

    
}
