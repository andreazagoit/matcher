import { gql } from "graphql-tag";

// ============================================
// FRAGMENTS
// ============================================

export const EVENT_CARD_FRAGMENT = gql`
  fragment EventCardFields on Event {
    id
    title
    description
    location
    cover
    startsAt
    endsAt
    maxAttendees
    attendeeCount
    myAttendeeStatus
    categories
    spaceId
    price
    isPaid
  }
`;

// ============================================
// QUERIES
// ============================================

export const GET_SPACE_EVENTS = gql`
  ${EVENT_CARD_FRAGMENT}
  query SpaceEvents($spaceId: ID!) {
    space(id: $spaceId) {
      id
      events {
        ...EventCardFields
        createdBy {
          id
          name
        }
        currency
      }
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
      cover
      images
      maxAttendees
      attendeeCount
      myAttendeeStatus
      myPaymentStatus
      categories
      spaceId
      createdBy {
        id
        name
        username
      }
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

export const GET_MY_UPCOMING_EVENTS = gql`
  ${EVENT_CARD_FRAGMENT}
  query MyUpcomingEvents {
    myUpcomingEvents {
      ...EventCardFields
    }
  }
`;

export const GET_ALL_EVENTS = gql`
  ${EVENT_CARD_FRAGMENT}
  query GetAllEvents($limit: Int, $offset: Int) {
    events(limit: $limit, offset: $offset) {
      ...EventCardFields
    }
  }
`;

// ============================================
// MUTATIONS
// ============================================

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
      categories
      price
      currency
    }
  }
`;

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

export const GET_EVENT_RECOMMENDED_EVENTS = gql`
  ${EVENT_CARD_FRAGMENT}
  query GetEventRecommendedEvents($id: ID!, $limit: Int) {
    event(id: $id) {
      id
      recommendedEvents(limit: $limit) {
        ...EventCardFields
        space {
          id
          name
          slug
        }
      }
    }
  }
`;
