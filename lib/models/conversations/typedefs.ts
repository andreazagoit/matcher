export const conversationTypeDefs = `#graphql
  enum ConversationStatus {
    request
    active
    declined
  }

  type Conversation {
    id: ID!
    initiator: User!
    recipient: User!
    otherUser: User!
    status: ConversationStatus!
    source: String
    lastMessage: Message
    lastMessageAt: DateTime
    unreadCount: Int
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  type Message {
    id: ID!
    conversationId: ID!
    sender: User!
    content: String!
    readAt: DateTime
    createdAt: DateTime!
  }

  extend type Query {
    """
    Get pending message requests (inbox).
    """
    messageRequests: [Conversation!]!

    """
    Get active conversations for the authenticated user.
    """
    conversations: [Conversation!]!

    """
    Get a single conversation by ID.
    """
    conversation(id: ID!): Conversation

    """
    Get messages for a conversation.
    """
    messages(conversationId: ID!): [Message!]!
  }

  extend type Mutation {
    """
    Send a message request to another user. Creates a conversation with status=request.
    """
    sendMessageRequest(recipientId: ID!, content: String!, source: String): Conversation!

    """
    Accept or decline a message request.
    """
    respondToRequest(conversationId: ID!, accept: Boolean!): Conversation!

    """
    Send a message in an existing conversation.
    """
    sendMessage(conversationId: ID!, content: String!): Message!

    """
    Mark all messages in a conversation as read.
    """
    markAsRead(conversationId: ID!): Boolean
  }
`;
