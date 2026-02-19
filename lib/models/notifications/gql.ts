import { gql } from "graphql-tag";

export const NOTIFICATION_FRAGMENT = gql`
  fragment NotificationFields on Notification {
    id
    type
    text
    image
    href
    read
    createdAt
  }
`;

export const GET_NOTIFICATIONS = gql`
  ${NOTIFICATION_FRAGMENT}
  query GetNotifications($limit: Int, $offset: Int) {
    notifications(limit: $limit, offset: $offset) {
      ...NotificationFields
    }
  }
`;

export const GET_UNREAD_COUNT = gql`
  query GetUnreadNotificationsCount {
    unreadNotificationsCount
  }
`;

export const MARK_NOTIFICATION_READ = gql`
  ${NOTIFICATION_FRAGMENT}
  mutation MarkNotificationRead($id: ID!) {
    markNotificationRead(id: $id) {
      ...NotificationFields
    }
  }
`;

export const MARK_ALL_NOTIFICATIONS_READ = gql`
  mutation MarkAllNotificationsRead {
    markAllNotificationsRead
  }
`;

export const DELETE_NOTIFICATION = gql`
  mutation DeleteNotification($id: ID!) {
    deleteNotification(id: $id)
  }
`;
