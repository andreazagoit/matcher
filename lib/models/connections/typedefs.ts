export const connectionTypeDefs = `#graphql
  enum ConnectionStatus {
    pending
    accepted
    declined
  }

  type Connection {
    id: ID!
    initiator: User!
    recipient: User!
    otherUser: User!
    status: ConnectionStatus!
    targetUserItem: UserItem!
    initialMessage: String
    lastMessage: Message
    lastMessageAt: DateTime
    unreadCount: Int
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  type Message {
    id: ID!
    connectionId: ID!
    sender: User!
    content: String!
    readAt: DateTime
    createdAt: DateTime!
  }

  extend type Query {
    """
    Get pending connection requests (inbox).
    """
    connectionRequests: [Connection!]!

    """
    Get accepted connections (Friends/Matches) for the authenticated user.
    """
    connections: [Connection!]!

    """
    Get a single connection by ID.
    """
    connection(id: ID!): Connection

    """
    Get messages for a connection.
    """
    messages(connectionId: ID!): [Message!]!
  }

  extend type Mutation {
    """
    Send a connection request to another user by liking/commenting on their profile item.
    """
    sendConnectionRequest(recipientId: ID!, targetUserItemId: ID!, initialMessage: String): Connection!

    """
    Accept or decline a connection request.
    """
    respondToRequest(connectionId: ID!, accept: Boolean!): Connection!

    """
    Send a message in an existing connection.
    """
    sendMessage(connectionId: ID!, content: String!): Message!

    """
    Mark all messages in a connection as read.
    """
    markAsRead(connectionId: ID!): Boolean
  }
`;
