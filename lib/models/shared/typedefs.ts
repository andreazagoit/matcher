export const sharedTypeDefs = `#graphql
  type Coordinates {
    lat: Float!
    lon: Float!
  }

  input CoordinatesInput {
    lat: Float!
    lon: Float!
  }

  type SpaceConnection {
    nodes: [Space!]!
    hasNextPage: Boolean!
  }

  type EventConnection {
    nodes: [Event!]!
    hasNextPage: Boolean!
  }
`;
