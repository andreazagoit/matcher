import { gql } from "graphql-tag";

// ============================================
// FRAGMENTS
// ============================================

export const CONVERSATION_FRAGMENT = gql`
  fragment ConversationFields on Conversation {
    id
    lastMessageAt
    createdAt
    updatedAt
    otherParticipant {
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

export const GET_RECENT_CONVERSATIONS = gql`
  query GetRecentConversations {
    conversations {
      id
      otherParticipant {
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
      otherParticipant {
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

export const START_CONVERSATION = gql`
  ${CONVERSATION_FRAGMENT}
  mutation StartConversation($targetUserId: ID!) {
    startConversation(targetUserId: $targetUserId) {
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
