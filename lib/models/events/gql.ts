import { gql } from "graphql-tag";

// ============================================
// QUERIES
// ============================================

export const GET_SPACE_EVENTS = gql`
  query SpaceEvents($spaceId: ID!) {
    spaceEvents(spaceId: $spaceId) {
      id
      title
      description
      location
      startsAt
      endsAt
      maxAttendees

      attendeeCount
      createdBy
      tags
      price
      currency
      isPaid
    }
  }
`;

export const GET_EVENT = gql`
  query GetEvent($id: ID!) {
    event(id: $id) {
      id
      title
      description
      location
      coordinates { lat lon }
      startsAt
      endsAt
      maxAttendees

      attendeeCount
      myAttendeeStatus
      myPaymentStatus
      tags
      spaceId
      createdBy
      createdAt
      price
      currency
      isPaid
      space {
        id
        name
        slug
        visibility
        stripeAccountEnabled
        myMembership {
          role
        }
      }
      attendees {
        id
        userId
        status
        registeredAt
        paymentStatus
        user {
          id
          name
          username
        }
      }
    }
  }
`;

export const UPDATE_EVENT = gql`
  mutation UpdateEvent($id: ID!, $input: UpdateEventInput!) {
    updateEvent(id: $id, input: $input) {
      id
      title
      description
      location
      startsAt
      endsAt
      maxAttendees

      tags
      price
      currency
    }
  }
`;

export const GET_MY_UPCOMING_EVENTS = gql`
  query MyUpcomingEvents {
    myUpcomingEvents {
      id
      title
      description
      location
      coordinates { lat lon }
      startsAt
      endsAt
      maxAttendees

      attendeeCount
      tags
      spaceId
    }
  }
`;

export const GET_RECOMMENDED_EVENTS = gql`
  query RecommendedEvents($limit: Int) {
    recommendedEvents(limit: $limit) {
      id
      title
      description
      location
      coordinates { lat lon }
      startsAt
      endsAt
      maxAttendees

      attendeeCount
      tags
      spaceId
    }
  }
`;

// ============================================
// MUTATIONS
// ============================================

export const CREATE_EVENT = gql`
  mutation CreateEvent($input: CreateEventInput!) {
    createEvent(input: $input) {
      id
    }
  }
`;

export const RESPOND_TO_EVENT = gql`
  mutation RespondToEvent($eventId: ID!, $status: AttendeeStatus!) {
    respondToEvent(eventId: $eventId, status: $status) {
      id
      status
    }
  }
`;

export const MARK_EVENT_COMPLETED = gql`
  mutation MarkEventCompleted($eventId: ID!) {
    markEventCompleted(eventId: $eventId) {
      id

    }
  }
`;
