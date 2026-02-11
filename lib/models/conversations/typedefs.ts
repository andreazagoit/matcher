export const conversationTypeDefs = `#graphql
  type Conversation {
    id: ID!
    participant1: User!
    participant2: User!
    lastMessageAt: String
    createdAt: String!
    updatedAt: String!
    
    # Computed fields
    otherParticipant: User!
    lastMessage: Message
    unreadCount: Int
  }

  type Message {
    id: ID!
    conversationId: ID!
    sender: User!
    content: String!
    readAt: String
    createdAt: String!
  }

  extend type Query {
    conversations: [Conversation!]!
    messages(conversationId: ID!): [Message!]!
    conversation(id: ID!): Conversation
  }

  extend type Mutation {
    startConversation(targetUserId: ID!): Conversation!
    sendMessage(conversationId: ID!, content: String!): Message!
    markAsRead(conversationId: ID!): Boolean
  }
`;
