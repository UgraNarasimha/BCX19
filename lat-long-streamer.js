const StreamrClient = require('streamr-client')

// Create the client and supply either an API key or an Ethereum private key to authenticate
const client = new StreamrClient({
    auth: {
        apiKey: 'paste your api key here',
        // Or to cryptographically authenticate with Ethereum and enable data signing:
        // privateKey: 'ETHEREUM-PRIVATE-KEY',
    },
})

// Create a stream for this example if it doesn't exist
client.getOrCreateStream({
    name: 'this-is-my-stream-name-for-my-streamer',
}).then((stream) => setInterval(() => {
    // Generate a message payload with a random number
    const msg = {,
		lat: 52.500220,
		lon: 13.375080,

    }

    // Publish the message to the Stream
    stream.publish(msg)
        .then(() => console.log('Sent successfully: ', msg))
        .catch((err) => console.error(err))
}, 1000))
