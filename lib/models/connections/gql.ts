import { gql } from "graphql-tag";

// ============================================
// FRAGMENTS
// ============================================

export const CONNECTION_FRAGMENT = gql`
  fragment ConnectionFields on Connection {
    id
    status
    targetUserItem {
      id
      content
      type
    }
    initialMessage
    lastMessageAt
    createdAt
    updatedAt
    otherUser {
      id
      name
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
    connectionId
    content
    readAt
    createdAt
    sender {
      id
      name
      image
    }
  }
`;

// ============================================
// QUERIES
// ============================================

export const GET_CONVERSATIONS = gql`
  ${CONNECTION_FRAGMENT}
  query GetConnections {
    connections {
      ...ConnectionFields
    }
  }
`;

export const GET_CONNECTION_REQUESTS = gql`
  ${CONNECTION_FRAGMENT}
  query GetConnectionRequests {
    connectionRequests {
      ...ConnectionFields
    }
  }
`;

export const GET_RECENT_CONVERSATIONS = gql`
  query GetRecentConnections {
    connections {
      id
      otherUser {
        name
      }
      unreadCount
    }
  }
`;

export const GET_MESSAGES = gql`
  ${MESSAGE_FRAGMENT}
  query GetMessages($connectionId: ID!) {
    messages(connectionId: $connectionId) {
      ...MessageFields
    }
    connection(id: $connectionId) {
      id
      status
      otherUser {
        id
        name
        image
      }
    }
  }
`;

// ============================================
// MUTATIONS
// ============================================

export const SEND_CONNECTION_REQUEST = gql`
  ${CONNECTION_FRAGMENT}
  mutation SendConnectionRequest($recipientId: ID!, $targetUserItemId: ID!, $initialMessage: String) {
    sendConnectionRequest(recipientId: $recipientId, targetUserItemId: $targetUserItemId, initialMessage: $initialMessage) {
      ...ConnectionFields
    }
  }
`;

export const RESPOND_TO_REQUEST = gql`
  ${CONNECTION_FRAGMENT}
  mutation RespondToRequest($connectionId: ID!, $accept: Boolean!) {
    respondToRequest(connectionId: $connectionId, accept: $accept) {
      ...ConnectionFields
    }
  }
`;

export const SEND_MESSAGE = gql`
  mutation SendMessage($connectionId: ID!, $content: String!) {
    sendMessage(connectionId: $connectionId, content: $content) {
      id
      content
      createdAt
    }
  }
`;

export const MARK_AS_READ = gql`
  mutation MarkAsRead($connectionId: ID!) {
    markAsRead(connectionId: $connectionId)
  }
`;

