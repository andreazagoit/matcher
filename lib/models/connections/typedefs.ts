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
    """Messages in this connection, newest first."""
    messages: [Message!]!
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
    """Accepted connections (chats) for the authenticated viewer."""
    myConnections: [Connection!]!
    """Pending incoming connection requests for the authenticated viewer."""
    myConnectionRequests: [Connection!]!
    """Single connection by ID (must belong to the authenticated viewer)."""
    connection(id: ID!): Connection
  }

  extend type Mutation {
    """Send a connection request to another user."""
    sendConnectionRequest(recipientId: ID!, targetUserItemId: ID!, initialMessage: String): Connection!
    """Accept or decline a connection request."""
    respondToRequest(connectionId: ID!, accept: Boolean!): Connection!
    """Send a message in an existing connection."""
    sendMessage(connectionId: ID!, content: String!): Message!
    """Mark all messages in a connection as read."""
    markAsRead(connectionId: ID!): Boolean
  }
`;
