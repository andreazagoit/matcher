import { gql } from "graphql-tag";

// ============================================
// FRAGMENTS
// ============================================

export const CONVERSATION_FRAGMENT = gql`
  fragment ConversationFields on Conversation {
    id
    status
    source
    lastMessageAt
    createdAt
    updatedAt
    otherUser {
      id
      givenName
      familyName
      image
    }
    lastMessage {
      content
      createdAt
    }
    unreadCount
  }
`;

export const MESSAGE_FRAGMENT = gql`
  fragment MessageFields on Message {
    id
    conversationId
    content
    readAt
    createdAt
    sender {
      id
      givenName
      familyName
      image
    }
  }
`;

// ============================================
// QUERIES
// ============================================

export const GET_CONVERSATIONS = gql`
  ${CONVERSATION_FRAGMENT}
  query GetConversations {
    conversations {
      ...ConversationFields
    }
  }
`;

export const GET_MESSAGE_REQUESTS = gql`
  ${CONVERSATION_FRAGMENT}
  query GetMessageRequests {
    messageRequests {
      ...ConversationFields
    }
  }
`;

export const GET_RECENT_CONVERSATIONS = gql`
  query GetRecentConversations {
    conversations {
      id
      otherUser {
        givenName
        familyName
      }
      unreadCount
    }
  }
`;

export const GET_MESSAGES = gql`
  ${MESSAGE_FRAGMENT}
  query GetMessages($conversationId: ID!) {
    messages(conversationId: $conversationId) {
      ...MessageFields
    }
    conversation(id: $conversationId) {
      id
      status
      otherUser {
        id
        givenName
        familyName
        image
      }
    }
  }
`;

// ============================================
// MUTATIONS
// ============================================

export const SEND_MESSAGE_REQUEST = gql`
  ${CONVERSATION_FRAGMENT}
  mutation SendMessageRequest($recipientId: ID!, $content: String!, $source: String) {
    sendMessageRequest(recipientId: $recipientId, content: $content, source: $source) {
      ...ConversationFields
    }
  }
`;

export const RESPOND_TO_REQUEST = gql`
  ${CONVERSATION_FRAGMENT}
  mutation RespondToRequest($conversationId: ID!, $accept: Boolean!) {
    respondToRequest(conversationId: $conversationId, accept: $accept) {
      ...ConversationFields
    }
  }
`;

export const SEND_MESSAGE = gql`
  mutation SendMessage($conversationId: ID!, $content: String!) {
    sendMessage(conversationId: $conversationId, content: $content) {
      id
      content
      createdAt
    }
  }
`;

export const MARK_AS_READ = gql`
  mutation MarkAsRead($conversationId: ID!) {
    markAsRead(conversationId: $conversationId)
  }
`;
